import os
import math
import zipfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import nmrglue as ng
import plotly.graph_objects as go
import numpy as np
import json
import sqlite3
from scipy.optimize import curve_fit
import uuid

# In-memory store for uploaded NMR datasets, keyed by session UUID
_nmr_data_store = {}

def safe_float(v, fallback=0.0):
    """Return a JSON-safe float, replacing NaN/Inf with fallback."""
    try:
        f = float(v)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return fallback

def safe_list(lst, fallback=0.0):
    """Return a list of JSON-safe floats."""
    return [safe_float(v, fallback) for v in lst]

# SQLite Configuration
DB_FILE = 'nmr_diffusion.db'

def init_db():
    try:
        # Create database and table if not exist
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS standards (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                diffusion_constant REAL NOT NULL,
                unit TEXT DEFAULT 'm^2/s'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibrations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                name TEXT NOT NULL,
                max_g_gauss REAL NOT NULL,
                delta REAL,
                big_delta REAL,
                fit_slope REAL,
                fit_intercept REAL
            )
        """)
        
        # Default standards data:
        # D2O: 1.9e-9 m^2/s
        # Glycerol: 2.2e-12 m^2/s (approx at RT)
        # Squalane: 3.1e-11 m^2/s (approx at RT)
        defaults = [
            ("D2O", 1.9e-9),
            ("Glycerol", 2.2e-12),
            ("Squalane", 3.1e-11)
        ]
        
        for name, d_const in defaults:
            cursor.execute("INSERT OR IGNORE INTO standards (name, diffusion_constant) VALUES (?, ?)", (name, d_const))
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not initialize database: {e}")

init_db()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

@app.route('/analyze_diffusion', methods=['POST'])
def analyze_diffusion():
    try:
        data = request.json
        peaks_ppm = data.get('peaks', [])
        method = data.get('method', 'intensity')
        calibration_id = data.get('calibration_id')
        data_id = data.get('data_id')
        if data_id and data_id in _nmr_data_store:
            plot_data = _nmr_data_store[data_id]
        else:
            plot_data = data.get('nmr_data', {})
        omitted_slices = data.get('omitted_slices', [])
        
        # 1. Fetch Calibration from DB
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cal = conn.execute("SELECT * FROM calibrations WHERE id = ?", (calibration_id,)).fetchone()
        conn.close()
        
        if not cal:
            return jsonify({'error': 'Calibration profile not found.'}), 404

        # 2. Extract Experimental Parameters from the Dataset
        # Priority: Parameters extracted from the uploaded files (acqus, etc.)
        exp_params = plot_data.get('exp_params', {})
        
        # Pull pulse program to determine the sequence and appropriate equation
        pulse_program = exp_params.get('pulse_program', 'unknown').lower()
        
        # Little delta (total gradient duration) and Big Delta (diffusion time)
        # We default to calibration values IF they are not found in the dataset,
        # but the DATASET values should override calibration for unknown samples.
        delta = exp_params.get('delta', cal['delta'])
        big_delta = exp_params.get('big_delta', cal['big_delta'])
        
        # Calibration constants (Slope/Intercept)
        gamma = 2.67522e8 # 1H gyromagnetic ratio (rad/s/T)
        m = cal['fit_slope']
        c = cal['fit_intercept']

        def lorentzian(x, amp, cen, wid, offset):
            """Lorentzian: wid = HWHM"""
            return amp * wid**2 / ((x - cen)**2 + wid**2) + offset

        def gaussian(x, amp, cen, sig, offset):
            """Gaussian: sig = sigma"""
            return amp * np.exp(-0.5 * ((x - cen) / sig)**2) + offset

        def pseudo_voigt(x, amp, cen, wid, eta, offset):
            """Pseudo-Voigt: linear mix of Lorentzian and Gaussian.
            eta = 0 → pure Gaussian, eta = 1 → pure Lorentzian"""
            lor = wid**2 / ((x - cen)**2 + wid**2)
            gau = np.exp(-0.5 * ((x - cen) / wid)**2)
            return amp * (eta * lor + (1 - eta) * gau) + offset

        def r_squared(y_data, y_fit):
            ss_res = np.sum((y_data - y_fit)**2)
            ss_tot = np.sum((y_data - np.mean(y_data))**2)
            return float(1 - ss_res / ss_tot) if ss_tot > 1e-20 else 0.0

        results_list = []
        # Support both 'ppm' and 'ppm_full' keys if they differ in plot_data
        # Revised check: process_nmr_data returns 'raw_ppm'
        ppm_array = np.array(plot_data.get('raw_ppm', plot_data.get('ppm', plot_data.get('ppm_full', []))))
        if len(ppm_array) == 0:
            # Fallback if the data structure is different
            return jsonify({'error': 'Data structure mismatch: PPM array not found.'}), 500
        
        full_ppm = ppm_array
        
        # Extract raw spectra to avoid including stackplot offsets (y_offset) in the fit
        raw_spectra = np.array(plot_data.get('raw_spectra', []))
        if len(raw_spectra) == 0:
            return jsonify({'error': 'Raw spectral data missing for analysis.'}), 400

        # Calculate X values for fitting (GRADIENTS)
        # Gradient DAC settings come from 'difframp' as fractions (0.0 to 1.0)
        x_points = np.array(plot_data.get('difframp', []))
        if len(x_points) == 0:
             return jsonify({'error': 'Gradient settings missing in data.'}), 400
        
        # Calculate gradients in G/cm using the user's chosen calibration slope/intercept
        # Formula: G = slope * (fraction) + intercept
        gradients = (m * x_points + c)
        
        # Determine sequence-specific fitting equation
        # 1. Convert G/cm to T/m: 1 G/cm = 0.01 T/m
        g_tesla_m = gradients * 0.01
        
        # DEFAULT: Basic Stejskal-Tanner (Double Stimulated Echo / BPPG)
        # Equation: I = I0 * exp(-D * (gamma * g * delta)^2 * (Delta - delta/3 - tau/2...))
        # Note: 'delta' is already the total gradient duration from param extraction
        if 'stebpgp' in pulse_program or 'bpp' in pulse_program:
            # BPP-STE (Bipolar Pulse Pair)
            # For Bipolar, the factor is (gamma * G * delta)^2 * (Delta - delta/3 - tau/2)
            # But the 'effective' delta in some literature for Bruker is 2*p30.
            # Here we follow Bruker's standard: X = (gamma * g * delta)^2 * (Delta - delta/3.0)
            st_x = (gamma * g_tesla_m * delta)**2 * (big_delta - delta/3.0)
        elif 'led' in pulse_program:
             # LED (Longitudinal Eddy Current Delay)
             st_x = (gamma * g_tesla_m * delta)**2 * (big_delta - delta/3.0)
        else:
            # General ST equation
            st_x = (gamma * g_tesla_m * delta)**2 * (big_delta - delta/3.0)

        for target_ppm in peaks_ppm:
            # Find the best index in the PPM array
            idx = np.argmin(np.abs(full_ppm - target_ppm))
            
            intensities = []
            peak_fit_models = [] # To store fit data for UI deconvolution visualization

            for slice_idx in range(len(raw_spectra)):
                slice_y = np.array(raw_spectra[slice_idx])

                if method in ('intensity', 'intensity_max'):
                    # Max intensity within ±15 points of the target PPM
                    window = 15
                    start = max(0, idx - window)
                    end = min(len(slice_y), idx + window)
                    segment = slice_y[start:end]
                    val = float(np.max(segment)) if len(segment) > 0 else 0.0
                    peak_fit_models.append(None)

                elif method == 'intensity_exact':
                    # Intensity at the exact PPM index (no window search)
                    val = float(slice_y[idx]) if idx < len(slice_y) else 0.0
                    peak_fit_models.append(None)

                else:  # method in ('area', 'intensity_fit')
                    # --- MULTI-MODEL PEAK FITTING ---
                    # Dynamic window: walk from peak centre to 0.5% of peak height
                    slice_y_ref = np.array(raw_spectra[0])
                    peak_amp_ref = float(slice_y_ref[idx]) if float(slice_y_ref[idx]) > 0 else float(np.max(slice_y_ref))
                    threshold = peak_amp_ref * 0.005

                    left = idx
                    while left > 0 and slice_y_ref[left] > threshold:
                        left -= 1
                    right = idx
                    while right < len(slice_y_ref) - 1 and slice_y_ref[right] > threshold:
                        right += 1

                    padding = 25
                    start = max(0, left - padding)
                    end   = min(len(slice_y), right + padding)

                    x_seg = full_ppm[start:end]
                    y_seg = slice_y[start:end]

                    if len(y_seg) < 8:
                        val = 0
                        peak_fit_models.append(None)
                    else:
                        # Baseline estimate from window edges
                        n_edge   = max(3, len(y_seg) // 10)
                        baseline = float(np.median(np.concatenate([y_seg[:n_edge], y_seg[-n_edge:]])))
                        y_sub    = y_seg - baseline

                        amp_g    = float(np.max(y_sub))
                        cen_g    = float(full_ppm[idx])
                        half_max = amp_g / 2.0
                        above    = x_seg[y_sub >= half_max]
                        hwhm_g   = float(abs(above[-1] - above[0]) / 2.0) if len(above) >= 2 else float(abs(full_ppm[1] - full_ppm[0]) * 3)
                        hwhm_g   = max(hwhm_g, 1e-4)
                        sig_g    = hwhm_g / np.sqrt(2 * np.log(2))  # HWHM → sigma for Gaussian

                        fits = {}  # name → (popt, r2, area, fwhm, fn_name)

                        # --- Lorentzian ---
                        try:
                            popt_l, _ = curve_fit(
                                lorentzian, x_seg, y_seg,
                                p0=[amp_g, cen_g, hwhm_g, baseline],
                                bounds=([0, cen_g-0.3, 1e-5, -np.inf],
                                        [np.inf, cen_g+0.3, 2.0,  np.inf]),
                                maxfev=10000
                            )
                            r2_l = r_squared(y_seg, lorentzian(x_seg, *popt_l))
                            area_l = float(np.pi * popt_l[0] * popt_l[2])
                            fwhm_l = float(2 * popt_l[2])
                            fits['Lorentzian'] = (popt_l, r2_l, area_l, fwhm_l, 'lorentzian')
                        except Exception as e:
                            print(f"    Lorentzian failed slice {slice_idx}: {e}")

                        # --- Gaussian ---
                        try:
                            popt_g, _ = curve_fit(
                                gaussian, x_seg, y_seg,
                                p0=[amp_g, cen_g, sig_g, baseline],
                                bounds=([0, cen_g-0.3, 1e-5, -np.inf],
                                        [np.inf, cen_g+0.3, 2.0,  np.inf]),
                                maxfev=10000
                            )
                            r2_g = r_squared(y_seg, gaussian(x_seg, *popt_g))
                            area_g = float(popt_g[0] * popt_g[2] * np.sqrt(2 * np.pi))
                            fwhm_g = float(2 * np.sqrt(2 * np.log(2)) * popt_g[2])
                            fits['Gaussian'] = (popt_g, r2_g, area_g, fwhm_g, 'gaussian')
                        except Exception as e:
                            print(f"    Gaussian failed slice {slice_idx}: {e}")

                        # --- Pseudo-Voigt ---
                        try:
                            popt_v, _ = curve_fit(
                                pseudo_voigt, x_seg, y_seg,
                                p0=[amp_g, cen_g, hwhm_g, 0.5, baseline],
                                bounds=([0, cen_g-0.3, 1e-5, 0, -np.inf],
                                        [np.inf, cen_g+0.3, 2.0, 1,  np.inf]),
                                maxfev=15000
                            )
                            r2_v = r_squared(y_seg, pseudo_voigt(x_seg, *popt_v))
                            eta_v = popt_v[3]
                            area_v = float(eta_v * np.pi * popt_v[0] * popt_v[2] +
                                           (1 - eta_v) * popt_v[0] * popt_v[2] * np.sqrt(2 * np.pi))
                            fwhm_v = float(2 * popt_v[2])
                            fits['Pseudo-Voigt'] = (popt_v, r2_v, area_v, fwhm_v, 'pseudo_voigt')
                        except Exception as e:
                            print(f"    Pseudo-Voigt failed slice {slice_idx}: {e}")

                        if fits:
                            # Pick the model with highest R²
                            best_name = max(fits, key=lambda k: fits[k][1])
                            best_popt, best_r2, best_area, best_fwhm, best_fn = fits[best_name]

                            # intensity_fit → use fitted amplitude; area → use integrated area
                            val = float(best_popt[0]) if method == 'intensity_fit' else best_area
                            x_dense = np.linspace(x_seg[0], x_seg[-1], 600)

                            if best_fn == 'lorentzian':
                                y_dense = lorentzian(x_dense, *best_popt)
                            elif best_fn == 'gaussian':
                                y_dense = gaussian(x_dense, *best_popt)
                            else:
                                y_dense = pseudo_voigt(x_dense, *best_popt)

                            all_r2 = {k: round(v[1], 4) for k, v in fits.items()}

                            peak_fit_models.append({
                                'x': x_dense.tolist(),
                                'y_fit': y_dense.tolist(),
                                'fit_type': best_name,
                                'r2': round(best_r2, 4),
                                'area': round(best_area, 4),
                                'fwhm_ppm': round(best_fwhm, 5),
                                'center_ppm': round(float(best_popt[1]), 4),
                                'amplitude': round(float(best_popt[0]), 4),
                                'all_r2': all_r2
                            })
                        else:
                            # All models failed — fallback
                            trap_val = float(np.trapz(y_sub, x_seg))
                            val = amp_g if method == 'intensity_fit' else trap_val
                            peak_fit_models.append({
                                'fit_type': 'Trapezoid (fallback)',
                                'r2': None,
                                'area': round(trap_val, 4),
                                'fwhm_ppm': None,
                                'center_ppm': round(cen_g, 4),
                                'amplitude': round(amp_g, 4),
                                'all_r2': {}
                            })
                
                intensities.append(val)
            
            intensities = np.array(intensities)
            
            # Linear fit in log-space: ln(I/I0) = -D * ST_X
            # Normalization: I/I0
            # Apply Omissions
            mask = np.ones(len(intensities), dtype=bool)
            for oi in omitted_slices:
                if 0 <= oi < len(mask):
                    mask[oi] = False
            
            # Robust fitting: Only fit points where intensity is above noise floor (e.g. 2% of I0)
            I0 = intensities[0] if intensities[0] > 0 else 1.0
            norm_intensities_full = intensities / I0
            noise_floor = 0.005 # Lower noise floor for better fitting of weak signals
            
            # Ensure we don't take log of zero or negative numbers
            valid = (norm_intensities_full > noise_floor) & mask
            
            if np.sum(valid) < 3:
                # If too few points are above noise, pick the strongest points available
                # (but still respect the omission mask)
                strongest_indices = np.argsort(norm_intensities_full)[::-1]
                valid = np.zeros_like(norm_intensities_full, dtype=bool)
                count = 0
                for idx_strong in strongest_indices:
                    if not omitted_slices or idx_strong not in omitted_slices:
                        if norm_intensities_full[idx_strong] > 1e-10: # Absolute zero protection
                            valid[idx_strong] = True
                            count += 1
                        if count >= 3: break
                
            # Final sanity check for valid points
            if np.sum(valid) < 2:
                # Total fallback if everything is zero/noisy
                log_i = np.zeros(2)
                x_fit = np.array([0, 1])
                slope, intercept = 0, 0
            else:
                log_i = np.log(norm_intensities_full[valid])
                x_fit = st_x[valid]
                
                # ln(I/I0) = -D * ST_X + offset
                try:
                    slope, intercept = np.polyfit(x_fit, log_i, 1)
                except Exception as e:
                    print(f"Fitting error for peak {target_ppm}: {e}")
                    slope, intercept = 0, 0
            
            d_value = -slope
            
            # Calculate R-squared and proper error statistics
            if np.sum(valid) >= 2:
                log_fit_at_points = slope * x_fit + intercept
                ss_res = np.sum((log_i - log_fit_at_points)**2)
                mean_log_i = np.mean(log_i)
                ss_tot = np.sum((log_i - mean_log_i)**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
                
                # Estimate error using standard deviation of the slope
                n_points = np.sum(valid)
                denom = np.sum((x_fit - np.mean(x_fit))**2)
                if n_points > 2 and denom > 1e-20:
                    std_err = np.sqrt(ss_res / (n_points - 2)) / np.sqrt(denom)
                else:
                    std_err = 0.1
            else:
                r2 = 0
                std_err = 0.1

            error_pct = (std_err / abs(slope)) if slope != 0 else 1.0

            # Fit intensities for display (normalized 0-1) across the full range
            # Increase resolution to 500 points for a perfectly smooth curve
            # We map from G_min to G_max (physical gradient units)
            g_smooth = np.linspace(0, np.max(gradients), 500)
            g_tesla_m_smooth = g_smooth * 0.01
            st_x_smooth = (gamma * g_tesla_m_smooth * delta)**2 * (big_delta - delta/3.0)
            
            fit_intensities_smooth = np.exp(slope * st_x_smooth + intercept)
            
            # Also calculate the fit values at the actual data points for UI
            fit_intensities_at_points = np.exp(slope * st_x + intercept)

            results_list.append({
                'ppm': safe_float(target_ppm),
                'intensities': safe_list(norm_intensities_full.tolist()),
                'fit_intensities': safe_list(fit_intensities_at_points.tolist()),
                'fit_line': {
                    'x': safe_list(g_smooth.tolist()),
                    'y': safe_list(fit_intensities_smooth.tolist())
                },
                'gradients': safe_list(gradients.tolist()),
                'd_value': safe_float(d_value),
                'r_squared': safe_float(r2),
                'error_pct': safe_float(error_pct),
                'peak_fits': peak_fit_models
            })

        return jsonify({
            'results': results_list,
            'params': {
                'delta': delta,
                'big_delta': big_delta,
                'pulse_program': pulse_program,
                'calibration_name': cal['name']
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.zip'):
        # Save zip
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Extract zip
        extract_dir = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(filename)[0])
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # Try to process the data
        try:
            # Check for standard type if provided
            standard_type = request.form.get('standard_type')
            
            plot_data = process_nmr_data(extract_dir)
            
            # If it's a standard, maybe fetch its known D from DB for future use
            if standard_type:
                # Store standard name in the response for frontend tracking
                plot_data['standard_name'] = standard_type
            
            data_id = str(uuid.uuid4())
            _nmr_data_store[data_id] = plot_data
            return jsonify({'message': 'File successfully uploaded and extracted', 'plot_data': plot_data, 'data_id': data_id})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Failed to process NMR data: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type. Please upload a .zip file'}), 400

def find_bruker_or_varian(base_dir):
    """Find the actual data directory by looking for Bruker or Varian signatures."""
    for root, dirs, files in os.walk(base_dir):
        if 'fid' in files and 'acqus' in files:
            return 'bruker', root
        if 'ser' in files and 'acqus' in files:
            return 'bruker', root
        if 'fid' in files and 'procpar' in files:
            return 'varian', root
    return None, None

def process_nmr_data(extract_dir):
    vendor, data_path = find_bruker_or_varian(extract_dir)

    if not vendor:
        raise ValueError("Could not detect Bruker or Varian data structure in the uploaded zip.")

    params = {'delta': 0.002, 'big_delta': 0.050, 'gamma': 2.67522e8} # Defaults
    difflist = None
    difframp = None

    if vendor == 'bruker':
        # Read the raw Bruker data
        dic, data = ng.bruker.read(data_path)
        
        # Helper to parse indexed arrays like $P= (0..63)
        def get_bruker_item(dic, file_path, prefix, index):
            # Try to get from dic first (nmrglue usually parses these into lists)
            try:
                if prefix == 'P' and 'P' in dic['acqus']:
                    return float(dic['acqus']['P'][index])
                if prefix == 'D' and 'D' in dic['acqus']:
                    return float(dic['acqus']['D'][index])
            except:
                pass
            
            # Fallback to manual parsing if nmrglue didn't get it
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                target = f"##${prefix}="
                for i, line in enumerate(lines):
                    if target in line:
                        vals = []
                        j = i + 1
                        while j < len(lines) and not lines[j].startswith('##'):
                            vals.extend(lines[j].strip().split())
                            j += 1
                        if index < len(vals):
                            return float(vals[index])
            except:
                pass
            return None

        # Try to extract parameters from 'acqus' directly for indexed values
        try:
            acqus_path = os.path.join(data_path, 'acqus')
            if os.path.exists(acqus_path):
                # P[30] is the pulse duration for the gradient half-pulse in stebpgp1s
                val_p30 = get_bruker_item(dic, acqus_path, 'P', 30)
                if val_p30 is not None:
                    # Little delta (total gradient duration) = 2 * P[30]
                    params['delta'] = 2.0 * val_p30 / 1000000.0
                
                # D[20] is the diffusion time (Big Delta)
                val_d20 = get_bruker_item(dic, acqus_path, 'D', 20)
                if val_d20 is not None:
                    params['big_delta'] = val_d20
            # Check for difflist
            difflist_path = os.path.join(data_path, 'difflist')
            if os.path.exists(difflist_path):
                with open(difflist_path, 'r') as f:
                    difflist = [float(line.strip()) for line in f if line.strip()]
            
            # Check for difframp in lists/gp/
            difframp_path = os.path.join(data_path, 'lists', 'gp', 'difframp')
            if os.path.exists(difframp_path):
                with open(difframp_path, 'r') as f:
                    lines = f.readlines()
                start_read = False
                difframp = []
                for line in lines:
                    if "##XYDATA" in line:
                        start_read = True
                        continue
                    if "##END" in line:
                        break
                    if start_read and line.strip():
                        difframp.append(float(line.strip()))
        except Exception as e:
            print(f"Param extraction error: {e}")
            pass

        data = ng.bruker.remove_digital_filter(dic, data)
        
        if not np.iscomplexobj(data) and len(data.shape) == 2:
            data = data[::2] + 1j * data[1::2]
            
    elif vendor == 'varian':
        dic, data = ng.varian.read(data_path)
        # TODO: Varian param extraction if needed

    # Convert to complex if still not complex
    if not np.iscomplexobj(data):
        data = data.astype(complex)

    # data should now be (N_gradients, N_points)
    if len(data.shape) == 1:
        slices = [data]
    else:
        slices = [data[i] for i in range(data.shape[0])]

    # If no data found, assume linear placeholder
    if difflist is None:
        difflist = np.linspace(2, 95, len(slices)).tolist()
    if difframp is None:
        # If difflist exists, it's often similar to the ramp or the power
        # For now, if missing, use 0-1 normalized version of difflist or indices
        difframp = (np.array(difflist) / max(difflist)).tolist()

    # Detect Standard based on pulse program or folder name
    detected_standard = None
    if vendor == 'bruker':
        pulse_program = dic.get('acqus', {}).get('PULPROG', '')
        if not pulse_program and 'acqus' in dic:
            # Sometimes PULPROG is a list or requires precise access
            pulse_program = dic['acqus'].get('PULPROG', [''])[0]
        pulse_program = str(pulse_program).lower()
        
        # In a real scenario, we'd check more metadata or the user might name their folder 'd2o_...'
        folder_name = os.path.normpath(data_path).split(os.sep)
        folder_search = " ".join(folder_name).lower()

        # Extract Pulse Program name
        params['pulse_program'] = pulse_program if pulse_program else "Unknown"

        if 'd2o' in folder_search or 'd2o' in pulse_program:
            detected_standard = 'D2O'
        elif 'glycerol' in folder_search or 'glyc' in pulse_program:
            detected_standard = 'Glycerol'
        elif 'squalane' in folder_search or 'squal' in pulse_program:
            detected_standard = 'Squalane'

    # Calculate magnitude spectra for all slices
    processed_spectra = []
    for i, trace in enumerate(slices):
        # 1. Stronger windowing (Exponential, lb=15) to isolate the peak
        window = np.exp(-15 * np.arange(len(trace)) / len(trace))
        trace_win = trace * window

        # 2. FT with 4x Zero-filling
        n_fft = len(trace) * 4
        sp = np.fft.fftshift(np.fft.fft(trace_win, n_fft))
        processed_spectra.append(np.abs(sp))

    # Normalize to the maximum of the FIRST gradient slice
    norm_factor = np.max(processed_spectra[0]) if len(processed_spectra) > 0 and np.max(processed_spectra[0]) > 0 else 1.0

    # Create a consistent ppm base scale
    ppm_base = np.linspace(15, -5, len(processed_spectra[0]))

    traces = []
    for i, sp_mag in enumerate(processed_spectra):
        sp_norm = sp_mag / norm_factor
        
        # Vertical offset for stacking
        y_offset = i * 0.05
        
        # Horizontal shift (ppm) to create a staggered waterfall effect
        x_shift = i * 0.1
        x_ppm_staggered = (ppm_base - x_shift)

        # Labels for the plot: Use difframp (%) if available, else indices
        label_val = (difframp[i] * 100) if difframp is not None else difflist[i]

        traces.append({
            'x': x_ppm_staggered.tolist(),
            'y': (sp_norm + y_offset).tolist(),
            'customdata': ppm_base.tolist(), # Store original ppm here
            'type': 'scatter',
            'mode': 'lines',
            'name': f'Step {i+1} ({label_val:.2f}%)',
            'hovertemplate': 'PPM: %{customdata:.3f}<br>Intensity: %{y:.3f}<extra></extra>'
        })

    layout = {
        'title': f'NMR {vendor.capitalize()} Diffusion Data (Pseudo-2D Stacked Plot)',
        'xaxis': {'title': 'Chemical Shift (ppm)', 'autorange': 'reversed'},
        'yaxis': {'title': 'Normalized Intensity', 'showticklabels': False, 'range': [0, 1.1 + (len(traces) * 0.05)]},
        'showlegend': True,
        'clickmode': 'event+select',
        'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
    }

    # Create a separate trace for the first slice (for peak selection)
    selection_trace = {
        'x': ppm_base.tolist(),
        'y': (processed_spectra[0] / norm_factor).tolist(),
        'type': 'scatter',
        'mode': 'lines',
        'name': 'Slice 1 (Selection)'
    }

    return {
        'stacked_data': traces, 
        'stacked_layout': layout,
        'selection_data': [selection_trace],
        'raw_ppm': ppm_base.tolist(),
        'raw_spectra': [sp.tolist() for sp in processed_spectra],
        'difflist': difflist,
        'difframp': difframp,
        'exp_params': params,
        'detected_standard': detected_standard
    }

@app.route('/get_calibrations', methods=['GET'])
def get_calibrations():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, timestamp, max_g_gauss, delta, big_delta, fit_slope, fit_intercept FROM calibrations ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{
        'id': r[0], 'name': r[1], 'timestamp': r[2], 
        'max_g': r[3], 'delta': r[4], 'big_delta': r[5],
        'slope': r[6], 'intercept': r[7]
    } for r in rows])

@app.route('/get_standards', methods=['GET'])
def get_standards():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name, diffusion_constant FROM standards")
    data = cursor.fetchall()
    conn.close()
    return jsonify([{'name': row[0], 'd_const': row[1]} for row in data])

@app.route('/analyze_peak', methods=['POST'])
def analyze_peak():
    data = request.json
    ppm_clicked = data.get('ppm')
    raw_ppm = np.array(data.get('raw_ppm'))
    raw_spectra = np.array(data.get('raw_spectra'))
    standard_name = data.get('standard')
    exp_params = data.get('exp_params', {})
    diff_ramp = np.array(data.get('difframp', []))

    if not standard_name:
        return jsonify({'error': 'Please select a standard type before analysis'}), 400

    # 1. Find the index of the ppm_clicked in raw_ppm
    idx = np.argmin(np.abs(raw_ppm - ppm_clicked))

    # 2. Extract intensities across all slices for this index
    intensities = raw_spectra[:, idx]

    # 3. Get known D for the standard
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT diffusion_constant FROM standards WHERE name = ?", (standard_name,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return jsonify({'error': 'Standard not found in database'}), 404
    
    D_known = row[0] # m^2/s

    # 4. Perform fitting using Stejskal-Tanner
    GAMMA = exp_params.get('gamma', 2.67522e8)
    DELTA = exp_params.get('delta', 0.002)
    BIG_DELTA = exp_params.get('big_delta', 0.050)
    
    # If no diff_ramp provided, default to linear 0.02 to 0.95
    if len(diff_ramp) == 0:
        diff_ramp = np.linspace(0.02, 0.95, len(intensities))
    
    def stejskal_tanner(ramp_val, I0, g_scale):
        # I = I0 * exp(-D_known * (gamma * g * delta)^2 * (Delta - delta/3.0))
        # Note: In stebpgp1s, p30 is little delta * 0.5. 
        # So we use 2 * p30 (which is params['delta'] here) as the total gradient duration.
        g = ramp_val * g_scale
        b_value = (GAMMA * g * DELTA)**2 * (BIG_DELTA - DELTA/3.0)
        return I0 * np.exp(-D_known * b_value)

    try:
        # P0: I0 = max intensity, g_scale = 50 T/m (500 G/cm approx) 
        # Note: popping 0.5 T/m (50 G/cm) is more standard for small bore
        popt, _ = curve_fit(stejskal_tanner, diff_ramp, intensities, p0=[intensities[0], 0.5])
        
        I0_fit = popt[0]
        G_max_fit = popt[1] # Calibrated max gradient in T/m
        
        fit_intensities = stejskal_tanner(diff_ramp, *popt)
        
        # Convert T/m to Gauss/cm: 1 T/m = 100 Gauss/cm
        calculated_max_g_gauss = G_max_fit * 100.0
        
        # Resultant gradient strengths in Gauss/cm
        gradients_gauss = diff_ramp * calculated_max_g_gauss

        # Perform a linear fit on: actual_gradients = slope * setting_percentage + intercept
        # diff_ramp is 0..1 (fraction), gradients_gauss is G/cm
        # We fit: y = m*x + c where x is the fraction
        slope, intercept = np.polyfit(diff_ramp, gradients_gauss, 1)
        
    except Exception as e:
        return jsonify({'error': f'Fitting failed: {str(e)}'}), 500

    return jsonify({
        'ppm': ppm_clicked,
        'intensities': intensities.tolist(),
        'fit_intensities': fit_intensities.tolist(),
        'gradient_steps': diff_ramp.tolist(),
        'gradients': gradients_gauss.tolist(),
        'calibrated_max_g': calculated_max_g_gauss,
        'fit_slope': float(slope),
        'fit_intercept': float(intercept),
        'delta': DELTA,
        'big_delta': BIG_DELTA,
        'difflist': data.get('difflist', [])
    })

@app.route('/save_calibration', methods=['POST'])
def save_calibration():
    data = request.json
    name = data.get('name')
    max_g = data.get('max_g')
    delta = data.get('delta')
    big_delta = data.get('big_delta')
    slope = data.get('slope')
    intercept = data.get('intercept')

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO calibrations (name, max_g_gauss, delta, big_delta, fit_slope, fit_intercept)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, max_g, delta, big_delta, slope, intercept))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Calibration saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=3000, debug=True)
