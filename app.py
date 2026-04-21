import os
import zipfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import nmrglue as ng
import plotly.graph_objects as go
import numpy as np
import json
import sqlite3
from scipy.optimize import curve_fit

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
            
            return jsonify({'message': 'File successfully uploaded and extracted', 'plot_data': plot_data})
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
        def get_bruker_param(file_path, prefix, index):
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    if f"##${prefix}= (0.." in line:
                        # Values are on subsequent lines
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
                val_p30 = get_bruker_param(acqus_path, 'P', 30)
                if val_p30 is not None:
                    # Little delta (total gradient duration) = 2 * P[30]
                    params['delta'] = 2.0 * val_p30 / 1000000.0
                
                # D[20] is the diffusion time (Big Delta)
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
        'exp_params': params
    }

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
