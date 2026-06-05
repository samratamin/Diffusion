import os
import math
import zipfile
from flask import Flask, render_template, request, jsonify, send_file, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
import nmrglue as ng
import plotly.graph_objects as go
import numpy as np
import json
import sqlite3
from scipy.optimize import curve_fit, minimize_scalar, minimize
import uuid
import io
import hashlib
import secrets
from functools import wraps
from datetime import datetime
import threading
import shutil
import time
import re

# In-memory store for uploaded NMR datasets, keyed by session UUID.
# Capped at 10 entries (LRU eviction) to prevent OOM on busy servers.
from collections import OrderedDict

class _LRUStore(OrderedDict):
    """OrderedDict with a max-size cap; evicts oldest entry when full."""
    def __init__(self, maxsize=10):
        super().__init__()
        self._maxsize = maxsize
    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self._maxsize:
            oldest_key, oldest_val = next(iter(self.items()))
            # Explicitly delete numpy arrays before eviction to free RAM immediately
            for k in list(oldest_val.keys()):
                if k.startswith('_'):
                    del oldest_val[k]
            self.popitem(last=False)
    def __getitem__(self, key):
        val = super().__getitem__(key)
        self.move_to_end(key)
        return val

_nmr_data_store = _LRUStore(maxsize=50)

def safe_float(v, fallback=0.0):
    """Return a JSON-safe float, replacing NaN/Inf with fallback."""
    try:
        f = float(v)
        return fallback if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return fallback

def safe_list(lst, fallback=0.0):
    """Return a list of JSON-safe floats. Uses numpy for performance on large arrays."""
    a = np.asarray(lst, dtype=np.float64)
    a = np.where(np.isfinite(a), a, fallback)
    return a.tolist()

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
            
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Migration: add vendor column to calibrations if it doesn't exist yet
        try:
            cursor.execute("ALTER TABLE calibrations ADD COLUMN vendor TEXT DEFAULT 'bruker'")
        except Exception:
            pass  # Column already exists

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT NOT NULL,
                data_id TEXT NOT NULL PRIMARY KEY,
                dataset_name TEXT,
                created_at REAL NOT NULL,
                summary_json TEXT
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_user_sessions ON user_sessions(session_id)"
        )
        # Clean up sessions older than 3 days on startup
        cutoff = time.time() - 3 * 86400
        old_ids = conn.execute(
            "SELECT data_id FROM user_sessions WHERE created_at < ?", (cutoff,)
        ).fetchall()
        conn.execute("DELETE FROM user_sessions WHERE created_at < ?", (cutoff,))
        for (did,) in old_ids:
            old_dir = os.path.join('uploads', did)
            if os.path.isdir(old_dir):
                shutil.rmtree(old_dir, ignore_errors=True)

        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not initialize database: {e}")

init_db()


def _get_admin_setting(key):
    try:
        conn = sqlite3.connect(DB_FILE)
        row = conn.execute("SELECT value FROM admin_settings WHERE key=?", (key,)).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _set_admin_setting(key, value):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT OR REPLACE INTO admin_settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def _get_or_create_secret_key():
    key = _get_admin_setting('secret_key')
    if not key:
        key = secrets.token_hex(32)
        _set_admin_setting('secret_key', key)
    return key

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = _get_or_create_secret_key()

# ─── Admin Auth ──────────────────────────────────────────────────────────────

def _hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(32)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return h, salt


def _verify_admin_password(password):
    stored_hash = _get_admin_setting('password_hash')
    salt = _get_admin_setting('password_salt')
    if not stored_hash or not salt:
        return False  # No password configured yet
    h, _ = _hash_password(password, salt)
    return secrets.compare_digest(h, stored_hash)


def _admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect('/admin/login')
        return f(*args, **kwargs)
    return decorated


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    # Generate a per-session CSRF token
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(24)
    error = None
    if request.method == 'POST':
        # Validate CSRF token
        if request.form.get('csrf_token') != session.get('csrf_token'):
            error = 'Invalid request. Please try again.'
        else:
            password = request.form.get('password', '')
            if not _get_admin_setting('password_hash'):
                error = 'No admin password has been set. Run: python manage.py set-password'
            elif _verify_admin_password(password):
                session['admin_logged_in'] = True
                session['csrf_token'] = secrets.token_hex(24)  # rotate after login
                return redirect('/admin')
            else:
                error = 'Incorrect password.'
    return render_template('admin_login.html',
                           error=error,
                           csrf_token=session.get('csrf_token', ''))


@app.route('/admin/logout')
def admin_logout():
    session.clear()
    return redirect('/admin/login')


@app.route('/admin')
@_admin_required
def admin_panel():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cals = conn.execute(
        "SELECT id, name, timestamp, max_g_gauss, delta, big_delta, fit_slope, fit_intercept, vendor "
        "FROM calibrations ORDER BY timestamp DESC"
    ).fetchall()
    conn.close()
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(24)
    ui_settings = {
        'show_method_area':              _get_admin_setting('ui_show_method_area') or '1',
        'show_method_intensity_exact':   _get_admin_setting('ui_show_method_intensity_exact') or '1',
        'show_method_intensity_fit':     _get_admin_setting('ui_show_method_intensity_fit') or '1',
    }
    return render_template('admin.html',
                           calibrations=cals,
                           csrf_token=session['csrf_token'],
                           ui_settings=ui_settings)


@app.route('/admin/calibration/<int:cal_id>/delete', methods=['POST'])
@_admin_required
def admin_delete_calibration(cal_id):
    if request.form.get('csrf_token') != session.get('csrf_token'):
        flash('Invalid request token.', 'error')
        return redirect('/admin')
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute("SELECT name FROM calibrations WHERE id=?", (cal_id,)).fetchone()
    if row:
        conn.execute("DELETE FROM calibrations WHERE id=?", (cal_id,))
        conn.commit()
        flash(f'Calibration "{row[0]}" deleted.', 'success')
    else:
        flash('Calibration not found.', 'error')
    conn.close()
    return redirect('/admin')


@app.route('/admin/calibration/<int:cal_id>/vendor', methods=['POST'])
@_admin_required
def admin_update_calibration_vendor(cal_id):
    if request.form.get('csrf_token') != session.get('csrf_token'):
        return ('Invalid token', 403)
    vendor = request.form.get('vendor', 'bruker').strip().lower()
    if vendor not in ('bruker', 'varian'):
        return ('Invalid vendor', 400)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE calibrations SET vendor=? WHERE id=?", (vendor, cal_id))
    conn.commit()
    conn.close()
    return ('', 204)

@app.route('/admin/ui_settings', methods=['POST'])
@_admin_required
def admin_ui_settings():
    if request.form.get('csrf_token') != session.get('csrf_token'):
        flash('Invalid request token.', 'error')
        return redirect('/admin')
    for key in ['ui_show_method_area', 'ui_show_method_intensity_exact', 'ui_show_method_intensity_fit']:
        _set_admin_setting(key, '1' if request.form.get(key) else '0')
    flash('UI settings saved.', 'success')
    return redirect('/admin')


@app.route('/api/ui_settings')
def api_ui_settings():
    """Public endpoint — returns which measurement method options are visible."""
    return jsonify({
        'show_method_area':            _get_admin_setting('ui_show_method_area') != '0',
        'show_method_intensity_exact': _get_admin_setting('ui_show_method_intensity_exact') != '0',
        'show_method_intensity_fit':   _get_admin_setting('ui_show_method_intensity_fit') != '0',
    })


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

        # Realign ppm to match actual spectrum length in case n_fft != n_collected
        # (same fix as in phase_spectrum / apply_processing)
        if raw_spectra.shape[1] != len(full_ppm):
            full_ppm = np.linspace(full_ppm[0], full_ppm[-1], raw_spectra.shape[1])

        # Calculate X values for fitting (GRADIENTS)
        # Gradient DAC settings come from 'difframp' as fractions (0.0 to 1.0)
        x_points = np.array(plot_data.get('difframp', []))
        if len(x_points) == 0:
             return jsonify({'error': 'Gradient settings missing in data.'}), 400
        
        # Calculate gradients in G/cm using the user's chosen calibration slope/intercept
        # Formula: G = slope * (fraction) + intercept
        gradients = (m * x_points + c)
        
        # Get sequence type and gradient shape for equation selection
        sequence_type = exp_params.get('sequence_type', 'PGSE')
        gradient_shape = exp_params.get('gradient_shape', 'square')
        gradient_shape_factor = exp_params.get('gradient_shape_factor', 1.0)
        # τ: inter-bipolar-gradient delay (non-zero for dbppste/bppste sequences)
        tau_bipolar = exp_params.get('tau_bipolar', 0.0)
        
        # 1. Convert G/cm to T/m: 1 G/cm = 0.01 T/m
        g_tesla_m = gradients * 0.01
        
        # Apply gradient shape correction factor if using sinusoidal gradients
        # For sinusoidal gradients, effective g is lower by ~0.9069
        g_effective = g_tesla_m * gradient_shape_factor
        
        # Stejskal-Tanner encoding factor — unified equation for all sequence types:
        #   PGSE:          X = (γ·g·δ)² · (Δ - δ/3)
        #   PGSTE bipolar: X = (γ·g·δ)² · (Δ - δ/3 - τ/2)   [dbppste, bppste]
        # tau_bipolar is 0 for non-bipolar sequences, reducing to standard PGSE form.
        st_x = (gamma * g_effective * delta)**2 * (big_delta - delta/3.0 - tau_bipolar/2.0)

        for target_ppm in peaks_ppm:
            # Find the best index in the PPM array
            idx = np.argmin(np.abs(full_ppm - target_ppm))
            
            intensities = []
            peak_fit_models = [] # To store fit data for UI deconvolution visualization
            peak_markers = []    # Exact points used for UI marker overlay (x/y on spectrum)

            for slice_idx in range(len(raw_spectra)):
                slice_y = np.array(raw_spectra[slice_idx])
                marker_idx = idx

                if method in ('intensity', 'intensity_max'):
                    # Peak max near target: find local maxima and choose the one
                    # closest to selected ppm (prevents jumping to neighboring peaks).
                    window = 20
                    start = max(0, idx - window)
                    end = min(len(slice_y), idx + window + 1)
                    segment = slice_y[start:end]
                    x_seg = full_ppm[start:end]

                    if len(segment) > 0:
                        # Try local maxima first to keep the picked point tied to target_ppm.
                        peaks, _ = find_peaks(segment)
                        if len(peaks) > 0:
                            nearest_peak_idx = peaks[np.argmin(np.abs(x_seg[peaks] - target_ppm))]
                            marker_idx = start + int(nearest_peak_idx)
                            val = float(slice_y[marker_idx])
                        else:
                            # Fallback for flat/broad peaks where no discrete local max is found.
                            marker_idx = start + int(np.argmax(segment))
                            val = float(slice_y[marker_idx])
                    else:
                        val = 0.0
                        marker_idx = idx
                    peak_fit_models.append(None)

                elif method == 'intensity_exact':
                    # Intensity at the exact PPM index (no window search)
                    val = float(slice_y[idx]) if idx < len(slice_y) else 0.0
                    marker_idx = idx
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

                            # For fit/area methods, draw the marker at fitted center.
                            marker_idx = int(np.argmin(np.abs(full_ppm - float(best_popt[1]))))
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
                            marker_idx = idx
                
                intensities.append(val)
                marker_idx = max(0, min(marker_idx, len(slice_y) - 1))
                peak_markers.append({
                    'ppm': safe_float(full_ppm[marker_idx]),
                    'intensity': safe_float(slice_y[marker_idx])
                })
            
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
            # Apply gradient_shape_factor here to match the actual fit (which used g_effective = g_tesla_m * gradient_shape_factor)
            st_x_smooth = (gamma * g_tesla_m_smooth * gradient_shape_factor * delta)**2 * (big_delta - delta/3.0 - tau_bipolar/2.0)
            
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
                'peak_fits': peak_fit_models,
                'peak_markers': peak_markers
            })

        return jsonify({
            'results': results_list,
            'params': {
                'delta': delta,
                'big_delta': big_delta,
                'pulse_program': pulse_program,
                'calibration_name': cal['name'],
                'sequence_type': exp_params.get('sequence_type', 'PGSE'),
                'gradient_shape': exp_params.get('gradient_shape', 'square'),
                'gradient_shape_factor': exp_params.get('gradient_shape_factor', 1.0),
                'tau_bipolar': exp_params.get('tau_bipolar', 0.0)
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ─── Data Download Endpoint ───────────────────────────────────────────────

def _build_readme(results, params, calibration, vendor, pulse_program,
                  delta, big_delta, ph0, ph1, baseline_order,
                  peaks_ppm, method, detected_standard, calibration_name, lb=1.0, fft_points=None,
                  sequence_type='PGSE', gradient_shape='square', gradient_shape_factor=1.0,
                  tau_bipolar=0.0, exp_params_full=None):
    """Build a publication-ready README.txt string."""
    ep = exp_params_full or {}
    lines = []
    w = lines.append

    w("=" * 78)
    w("  NMR DIFFUSION ANALYSIS — EXPERIMENT & RESULTS SUMMARY")
    w("  Generated by Diffusion Analysis Tool")
    w(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 78)
    w("")

    # ── 1. Experimental Parameters ──
    w("1. EXPERIMENTAL PARAMETERS")
    w("-" * 78)
    w(f"  Vendor / Format:        {vendor.upper()}")

    sfo1 = ep.get('sfo1_mhz')
    if sfo1:
        freq_mhz = round(float(sfo1))
        w(f"  ¹H Frequency:           {freq_mhz} MHz  ({float(sfo1):.4f} MHz exact)")
    else:
        w(f"  ¹H Frequency:           N/A")

    sw_hz = ep.get('sw_hz')
    sw_ppm = ep.get('sw_ppm')
    if sw_hz and sw_ppm:
        w(f"  Spectral width:         {sw_hz/1000:.2f} kHz  ({sw_ppm:.2f} ppm)")
    elif sw_hz:
        w(f"  Spectral width:         {sw_hz/1000:.2f} kHz")

    n_scans = ep.get('n_scans')
    if n_scans:
        w(f"  Scans per step:         {n_scans}")

    recycle = ep.get('recycle_delay')
    if recycle:
        w(f"  Recycle delay (d1):     {float(recycle):.3f} s")

    w(f"  Pulse Program:          {pulse_program}")
    w(f"  Sequence Type:          {sequence_type} (Pulsed Gradient {'Spin Echo' if sequence_type == 'PGSE' else 'Stimulated Echo'})")
    w(f"  Gradient Shape:         {gradient_shape.capitalize()}")
    if gradient_shape_factor != 1.0:
        w(f"  Gradient Shape Factor:  {gradient_shape_factor:.4f} (correction applied to effective gradient)")
    w(f"  δ (gradient pulse):     {delta:.6f} s  ({delta*1e6:.2f} μs)")
    w(f"  Δ (diffusion time):     {big_delta:.6f} s  ({big_delta*1e3:.2f} ms)")
    if tau_bipolar > 0:
        w(f"  τ (inter-bipolar delay): {tau_bipolar:.6f} s  ({tau_bipolar*1e3:.3f} ms)")
    w(f"  Number of gradient steps: {len(results[0]['intensities']) if results else 'N/A'}")
    w("")

    # ── 2. Calibration ──
    w("2. GRADIENT CALIBRATION")
    w("-" * 78)
    if calibration:
        w(f"  Calibration name:       {calibration_name}")
        w(f"  Slope (m):              {calibration['slope']:.6f} G/cm per DAC unit")
        w(f"  Intercept (c):          {calibration['intercept']:.6f} G/cm")
        w(f"  Max gradient strength:  {calibration['max_g']:.2f} G/cm")
        w(f"  Calibration date:       {calibration.get('timestamp', 'N/A')}")
    else:
        w("  Calibration:            Not applied")
    w("")

    # ── 3. Processing Details ──
    w("3. DATA PROCESSING")
    w("-" * 78)
    w(f"  Phase correction (ph0): {ph0:.2f}°")
    w(f"  Phase correction (ph1): {ph1:.2f}°")
    w(f"  Baseline correction:    {'Polynomial order ' + str(baseline_order) if baseline_order >= 0 else 'None'}")
    
    # Calculate zero-fill factor for README
    n_collected = params.get('n_collected', 0) if params else 0
    if fft_points:
        zerofill_text = f"{fft_points:,} points"
    elif n_collected:
        zerofill_text = f"{n_collected:,} points"
    else:
        zerofill_text = "N/A"
    
    w(f"  Apodization:            Exponential (lb = {lb:.1f} Hz)")
    w(f"  Zero-filling:           {zerofill_text}")
    
    # Sequence-specific equation information
    if sequence_type == 'PGSTE' and tau_bipolar > 0:
        w(f"  Diffusion equation:     Bipolar PGSTE: X = (γ·g·δ)² · (Δ - δ/3 - τ/2)")
        w(f"                          τ = {tau_bipolar*1e3:.3f} ms (inter-bipolar delay, Bruker D[21])")
    elif sequence_type == 'PGSTE':
        w(f"  Diffusion equation:     PGSTE: X = (γ·g·δ)² · (Δ - δ/3)")
    else:
        w(f"  Diffusion equation:     PGSE: X = (γ·g·δ)² · (Δ - δ/3)")
    
    if gradient_shape == 'sinusoidal':
        w(f"  Gradient shape:         Sinusoidal")
        w(f"  Gradient correction:    Applied correction factor {gradient_shape_factor:.4f}")
    else:
        w(f"  Gradient shape:         Square")
    
    w(f"  Intensity extraction:   {method}")
    w(f"  Selected peaks:         {', '.join(f'{p:.3f} ppm' for p in peaks_ppm)}")
    w("")

    # ── 4. Results Summary ──
    w("4. DIFFUSION COEFFICIENT RESULTS")
    w("-" * 78)
    w(f"  {'Peak (ppm)':<14} {'D (m²/s)':<18} {'R²':<10} {'Error (%)':<12}")
    w(f"  {'-'*12:<14} {'-'*16:<18} {'-'*8:<10} {'-'*10:<12}")

    d_values = []
    for r in results:
        d = r['d_value']
        d_values.append(d)
        w(f"  {r['ppm']:<14.3f} {d:<18.3e} {r['r_squared']:<10.4f} {r['error_pct']:<12.2f}")

    w("")
    if d_values:
        mean_d = sum(d_values) / len(d_values)
        if len(d_values) > 1:
            std_d = (sum((d - mean_d)**2 for d in d_values) / (len(d_values) - 1))**0.5
            se_d = std_d / (len(d_values)**0.5)
        else:
            std_d = 0.0
            se_d = 0.0
        w(f"  Mean D:               {mean_d:.3e} m²/s  (± {std_d:.3e})")
        w(f"  SE of mean D:         {se_d:.3e} m²/s")
        w(f"  Number of peaks:      {len(d_values)}")
    w("")

    # ── 5. Gradient Table ──
    w("5. GRADIENT TABLE")
    w("-" * 78)
    w(f"  {'Slice':<8} {'DAC (%)':<12} {'G (G/cm)':<14}")
    w(f"  {'-'*6:<8} {'-'*10:<12} {'-'*12:<14}")

    if results and results[0].get('gradients'):
        grads = results[0]['gradients']
        for i, g in enumerate(grads):
            w(f"  {i+1:<8} {g:<12.4f} {g:<14.4f}")
    w("")

    # ── 6. Notes ──
    w("6. NOTES")
    w("-" * 78)
    w("  - D values are extracted from the slope of ln(I/I₀) vs. ST_X using")
    w("    the Stejskal-Tanner equation: I = I₀·exp(-D·ST_X)")
    if tau_bipolar > 0:
        w(f"  - ST_X = (γ·g·δ)²·(Δ - δ/3 - τ/2)  bipolar correction with τ = {tau_bipolar*1e3:.3f} ms")
    else:
        w("  - ST_X = (γ·g·δ)²·(Δ - δ/3)  where γ = 2.67522×10⁸ rad/s/T (¹H gyromagnetic ratio)")
    w("  - Intensities are normalized to I₀ (first slice, lowest gradient).")
    w("  - Points below 0.5% of I₀ are excluded from the fit (noise floor).")
    w("")
    w("=" * 78)
    w("  End of report.")
    w("=" * 78)

    return "\n".join(lines)


def _build_publication_info(results, params, calibration, vendor, pulse_program,
                            delta, big_delta, ph0, ph1, baseline_order,
                            peaks_ppm, method, detected_standard, calibration_name,
                            lb=1.0, fft_points=None, exp_params_full=None,
                            sequence_type='PGSE', tau_bipolar=0.0):
    """Build a publication-ready Methods blurb with all available parameters filled in."""
    if not results:
        return "No analysis results available."

    ep = exp_params_full or {}

    d_values = [r['d_value'] for r in results]
    mean_d = sum(d_values) / len(d_values)
    n_peaks = len(d_values)
    if n_peaks > 1:
        std_d = (sum((d - mean_d)**2 for d in d_values) / (n_peaks - 1))**0.5
    else:
        std_d = 0.0
    avg_r2 = sum(r['r_squared'] for r in results) / len(results)

    # Gradient info
    grads = results[0].get('gradients', [])
    g_min = min(grads) if grads else 0.0
    g_max = max(grads) if grads else 0.0
    n_steps = len(grads) if grads else 0

    # Spectrometer / acquisition fields
    sfo1 = ep.get('sfo1_mhz')
    freq_str = f"{round(float(sfo1))} MHz" if sfo1 else "[frequency — check instrument]"
    vendor_cap = vendor.capitalize() if vendor and vendor != 'unknown' else "[Bruker/Varian]"

    sw_hz = ep.get('sw_hz')
    sw_ppm = ep.get('sw_ppm')
    if sw_hz and sw_ppm:
        sw_str = f"{sw_hz/1000:.1f} kHz ({sw_ppm:.1f} ppm)"
    elif sw_hz:
        sw_str = f"{sw_hz/1000:.1f} kHz"
    else:
        sw_str = "[spectral width — check instrument]"

    n_scans = ep.get('n_scans')
    scans_str = str(int(n_scans)) if n_scans else "[N]"

    recycle = ep.get('recycle_delay')
    recycle_str = f"{float(recycle):.1f} s" if recycle else "[d1 — check instrument]"

    # Processing
    n_collected = params.get('n_collected', 0) if params else 0
    if fft_points:
        pts_str = f"{fft_points:,}"
    elif n_collected:
        pts_str = f"{n_collected:,}"
    else:
        pts_str = "[N]"

    # Intensity method
    method_desc = {
        'intensity':       'peak maximum intensity',
        'intensity_max':   'peak maximum intensity',
        'intensity_exact': 'peak intensity at the exact chemical shift',
        'area':            'peak integrated area',
        'intensity_fit':   'fitted peak amplitude',
    }.get(method, method)

    baseline_desc = (f"polynomial order {baseline_order} baseline correction"
                     if baseline_order >= 0 else "no baseline correction")

    # Equation label
    if sequence_type == 'PGSTE' and tau_bipolar > 0:
        eq_label = "bipolar PGSTE Stejskal-Tanner equation"
        b_factor = f"(γgδ)²(Δ − δ/3 − τ/2)"
        tau_line = f"\nwhere τ = {tau_bipolar*1e3:.3f} ms is the inter-bipolar-gradient delay."
    else:
        eq_label = "Stejskal-Tanner equation"
        b_factor = "(γgδ)²(Δ − δ/3)"
        tau_line = ""

    # Calibration sentence
    if calibration and calibration.get('slope') is not None:
        icept = calibration.get('intercept', 0)
        sign = '+' if icept >= 0 else '-'
        cal_sentence = (f"The gradient strength was calibrated using {calibration_name}: "
                        f"G = {calibration['slope']:.5f}·s {sign} {abs(icept):.5f} G/cm "
                        f"(max {calibration['max_g']:.2f} G/cm).")
    else:
        cal_sentence = "Gradient calibration was applied prior to analysis."

    if detected_standard:
        std_sentence = f"The calibration standard was {detected_standard}."
    else:
        std_sentence = ""

    # Peak list
    if peaks_ppm:
        peak_list = ", ".join(f"{p:.3f} ppm" for p in peaks_ppm)
        peak_sentence = f"Peaks at {peak_list} were selected for fitting."
    else:
        peak_sentence = ""

    # Build blurb
    lines = []
    w = lines.append

    w("METHODS — COPY & PASTE TEMPLATE")
    w("=" * 78)
    w("(Items in [brackets] were not available from the data file and should")
    w(" be filled in manually before submission.)")
    w("")
    w(f"Diffusion coefficients were measured by pulsed-field gradient NMR")
    w(f"(PFG-NMR) on a {vendor_cap} {freq_str} spectrometer using the")
    w(f"{pulse_program} pulse sequence "
      f"({'bipolar PGSTE' if sequence_type=='PGSTE' and tau_bipolar>0 else sequence_type}). "
      f"Experiments were performed at [temperature] °C on [sample description].")
    w("")
    w(f"{n_steps} gradient steps were applied ranging from {g_min:.2f} to {g_max:.2f} G/cm,")
    w(f"with gradient pulse duration δ = {delta*1e6:.2f} μs and diffusion delay")
    w(f"Δ = {big_delta*1e3:.2f} ms{f', inter-bipolar delay τ = {tau_bipolar*1e3:.3f} ms' if tau_bipolar > 0 else ''}.")
    w(f"Each step was signal-averaged over {scans_str} scans with a {recycle_str} recycle delay.")
    w(f"Spectra were acquired with a spectral width of {sw_str}.")
    w(cal_sentence)
    if std_sentence:
        w(std_sentence)
    w("")
    w(f"Raw FIDs were apodized with an exponential window (lb = {lb:.1f} Hz),")
    w(f"Fourier-transformed to {pts_str} points, and phase-corrected")
    w(f"(ph0 = {ph0:.1f}°, ph1 = {ph1:.1f}°) with {baseline_desc}.")
    w(f"{peak_sentence}")
    w(f"The {method_desc} was extracted at each gradient step and fitted to")
    w(f"the {eq_label}:")
    w(f"")
    w(f"    I(g) = I₀ · exp(−D · {b_factor}){tau_line}")
    w(f"")
    w(f"where γ = 2.67522 × 10⁸ rad s⁻¹ T⁻¹ is the ¹H gyromagnetic ratio.")
    w(f"Intensities were normalized to I₀ (lowest-gradient slice).")
    w("")
    w(f"RESULTS SUMMARY")
    w("-" * 78)
    w(f"  Peaks analysed:   {n_peaks}  ({', '.join(f'{p:.3f} ppm' for p in peaks_ppm)})")
    w(f"  Mean D:           {mean_d:.3e} ± {std_d:.3e} m²/s  (mean ± SD)")
    w(f"  Mean R²:          {avg_r2:.4f}")
    for r in results:
        w(f"    {r['ppm']:.3f} ppm:  D = {r['d_value']:.3e} m²/s,  R² = {r['r_squared']:.4f},  err = {r['error_pct']:.1f}%")
    w("")
    w("NOTE: Fill in [temperature], [sample description], and probe type")
    w("before using in a manuscript.")
    w("")

    return "\n".join(lines)


# Gyromagnetic ratio constant (used in README generation)
gamma_const = 2.67522e8  # 1H gyromagnetic ratio (rad/s/T)


@app.route('/download_analysis', methods=['POST'])
def download_analysis():
    """Download all analysis data as a zipped package with README."""
    try:
        data = request.json
        results = data.get('results', [])
        params = data.get('params', {})
        calibration = data.get('calibration', {})
        peaks_ppm = data.get('peaks_ppm', [])
        method = data.get('method', 'intensity')
        detected_standard = data.get('detected_standard', None)
        ph0 = data.get('ph0', 0.0)
        ph1 = data.get('ph1', 0.0)
        baseline_order = data.get('baseline_order', -1)
        vendor = data.get('vendor', 'unknown')
        pulse_program = data.get('pulse_program', 'unknown')
        delta = data.get('delta', 0.0)
        big_delta = data.get('big_delta', 0.0)
        calibration_name = data.get('calibration_name', 'N/A')
        lb = data.get('lb', 1.0)
        fft_points = data.get('fft_points')
        if fft_points:
            fft_points = int(fft_points)
        sequence_type = data.get('sequence_type', 'PGSE')
        gradient_shape = data.get('gradient_shape', 'square')
        gradient_shape_factor = data.get('gradient_shape_factor', 1.0)
        tau_bipolar = float(data.get('tau_bipolar', 0.0))

        if not results:
            return jsonify({'error': 'No results to download.'}), 400

        # Inject n_collected into params so _build_readme can use it
        if params is None:
            params = {}
        params['n_collected'] = int(data.get('n_collected', 0))
        exp_params_full = data.get('exp_params', {}) or {}

        # ── Build README ──
        readme_text = _build_readme(
            results, params, calibration, vendor, pulse_program,
            delta, big_delta, ph0, ph1, baseline_order,
            peaks_ppm, method, detected_standard, calibration_name, lb, fft_points,
            sequence_type, gradient_shape, gradient_shape_factor,
            tau_bipolar, exp_params_full
        )

        # ── Build publication Methods blurb ──
        pub_info_text = _build_publication_info(
            results, params, calibration, vendor, pulse_program,
            delta, big_delta, ph0, ph1, baseline_order,
            peaks_ppm, method, detected_standard, calibration_name,
            lb, fft_points, exp_params_full,
            sequence_type, tau_bipolar
        )

        # ── Build CSV data for each peak ──
        csv_parts = []
        for r in results:
            csv_lines = []
            csv_lines.append("Peak_PPM,Intensity_Normalized,Fit_Intensity,Gradient_G_per_cm,ST_X")
            n = len(r['intensities'])
            for i in range(n):
                csv_lines.append(f"{r['ppm']},{r['intensities'][i]},{r['fit_intensities'][i]},{r['gradients'][i]},{r['fit_line']['x'][i] if i < len(r['fit_line']['x']) else ''}")
            csv_parts.append((f"peak_{r['ppm']:.3f}.csv", "\n".join(csv_lines)))

        # ── Build PPM axis + all raw spectra CSV ──
        ppm_array = data.get('raw_ppm', [])
        raw_spectra = data.get('raw_spectra', [])
        if ppm_array and raw_spectra:
            raw_csv_lines = []
            raw_csv_lines.append("PPM," + ",".join(f"Slice_{i+1}" for i in range(len(raw_spectra))))
            n_pts = len(ppm_array)
            for i in range(n_pts):
                row = [f"{ppm_array[i]:.6f}"]
                for j in range(len(raw_spectra)):
                    if i < len(raw_spectra[j]):
                        row.append(f"{raw_spectra[j][i]:.6f}")
                    else:
                        row.append("")
                raw_csv_lines.append(",".join(row))
            csv_parts.append(("all_spectra.csv", "\n".join(raw_csv_lines)))

        # ── Build decay data CSV (all peaks combined) ──
        decay_csv_lines = []
        decay_csv_lines.append("Gradient_G_per_cm," + ",".join(f"Peak_{r['ppm']:.3f}_I_norm" for r in results))
        n = len(results[0]['gradients']) if results else 0
        for i in range(n):
            row = [f"{results[0]['gradients'][i]:.6f}"]
            for r in results:
                if i < len(r['intensities']):
                    row.append(f"{r['intensities'][i]:.6f}")
                else:
                    row.append("")
            decay_csv_lines.append(",".join(row))
        csv_parts.append(("decay_data.csv", "\n".join(decay_csv_lines)))

        # ── Build fit CSV files (gradient vs fitted curve for each peak) ──
        fit_csv_parts = []
        for r in results:
            fit_csv_lines = []
            fit_csv_lines.append("Gradient_G_per_cm,Fitted_Intensity,ST_X")
            n = len(r['fit_line']['x'])
            for i in range(n):
                fit_csv_lines.append(f"{r['fit_line']['x'][i]:.6f},{r['fit_line']['y'][i]:.6f},{r['gradients'][i] if i < len(r['gradients']) else ''}")
            fit_csv_parts.append((f"fit_peak_{r['ppm']:.3f}.csv", "\n".join(fit_csv_lines)))

        # ── Generate plot images using matplotlib ──
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from io import BytesIO as Bio

        plot_images = {}
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # ── Plot 1: Decay curves matched to on-screen plot (linear axes) ──
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
        for idx, r in enumerate(results):
            color = colors[idx % len(colors)]
            gradients = np.array(r.get('gradients', []), dtype=float)
            intensities = np.array(r.get('intensities', []), dtype=float)
            if len(gradients) == 0 or len(intensities) == 0:
                continue

            ax.scatter(
                gradients,
                intensities,
                s=22,
                color=color,
                alpha=0.9,
                label=f"{float(r['ppm']):.3f} ppm",
                zorder=3,
            )

            if r.get('fit_line') and r['fit_line'].get('x') and r['fit_line'].get('y'):
                ax.plot(
                    np.array(r['fit_line']['x'], dtype=float),
                    np.array(r['fit_line']['y'], dtype=float),
                    color=color,
                    linewidth=2.0,
                    alpha=0.9,
                    zorder=2,
                )
            elif r.get('fit_intensities'):
                ax.plot(
                    gradients,
                    np.array(r['fit_intensities'], dtype=float),
                    color=color,
                    linewidth=1.5,
                    linestyle='--',
                    alpha=0.8,
                    zorder=2,
                )

        ax.set_xlabel('Gradient Strength (G/cm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Normalized Intensity (I/I0)', fontsize=11, fontweight='bold')
        ax.set_title('Decay Fit (Export Matched to UI)', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best', framealpha=0.9)
        ax.tick_params(axis='both', which='major', labelsize=9)
        plt.tight_layout()
        buf = Bio()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_images['decay_fit_ui_matched.png'] = buf.getvalue()

        # ── Plot 3: Angled stacked spectra (processed/phased data) ──
        if ppm_array and raw_spectra:
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
            ppm_np = np.array(ppm_array, dtype=float)
            first_slice = np.array(raw_spectra[0], dtype=float)
            norm_factor = float(np.max(np.abs(first_slice))) if len(first_slice) else 1.0
            if norm_factor == 0:
                norm_factor = 1.0

            for i, sp in enumerate(raw_spectra):
                sp_np = np.array(sp, dtype=float)
                if len(sp_np) != len(ppm_np):
                    continue
                y_offset = i * 0.05
                x_shift = i * 0.1
                x_plot = ppm_np - x_shift
                y_plot = (sp_np / norm_factor) + y_offset
                ax.plot(x_plot, y_plot, linewidth=1.1, color=colors[i % len(colors)], alpha=0.9)

            ax.set_xlabel('Chemical Shift (ppm)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Normalized Intensity + Offset', fontsize=11, fontweight='bold')
            ax.set_title('Angled Stacked Spectra (Processed/Phased)', fontsize=12, fontweight='bold', pad=10)
            ax.invert_xaxis()
            ax.grid(True, alpha=0.2)
            ax.tick_params(axis='both', which='major', labelsize=9)
            plt.tight_layout()
            buf = Bio()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            plot_images['stacked_spectra_angled.png'] = buf.getvalue()

        # ── Plot 4: Results summary bar chart (D values with error bars) ──
        if results:
            d_vals = [r['d_value'] for r in results]
            ppm_labels = [f"{r['ppm']:.2f}" for r in results]
            errors = [abs(r['d_value'] * r['error_pct']) for r in results]
            fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
            x_pos = range(len(d_vals))
            bars = ax.bar(x_pos, d_vals, yerr=errors, capsize=4,
                          color=[colors[i % len(colors)] for i in range(len(d_vals))],
                          alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Peak Position (ppm)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Diffusion Coefficient D (m²/s)', fontsize=11, fontweight='bold')
            ax.set_title('Diffusion Coefficients by Peak', fontsize=12, fontweight='bold', pad=10)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(ppm_labels, fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            buf = Bio()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            plot_images['diffusion_coefficients.png'] = buf.getvalue()

        # ── Plot 5: Calibration curve (if calibration data available) ──
        if calibration and calibration.get('slope') is not None:
            fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
            # Simulated calibration line
            g_range = np.linspace(0, calibration['max_g'] * 1.05, 100)
            slope = calibration['slope']
            intercept = calibration.get('intercept', 0)
            ax.plot(g_range, (g_range - intercept) / slope if slope != 0 else g_range,
                    color='blue', linewidth=2, label=f'G = {slope:.6f}·s {"" if intercept >= 0 else "-"}{abs(intercept):.6f}')
            ax.set_xlabel('DAC Setting (%)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Gradient Strength (G/cm)', fontsize=11, fontweight='bold')
            ax.set_title('Gradient Calibration Curve', fontsize=12, fontweight='bold', pad=10)
            ax.legend(fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=9)
            plt.tight_layout()
            buf = Bio()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            plot_images['calibration_curve.png'] = buf.getvalue()

        # ── Package into a zip ──
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README.txt", readme_text)
            zf.writestr("PublicationInfo.txt", pub_info_text)
            for fname, content in csv_parts:
                zf.writestr(fname, content)
            for fname, content in fit_csv_parts:
                zf.writestr(fname, content)
            for fname, img_data in plot_images.items():
                zf.writestr(f"plots/{fname}", img_data)

        zip_buffer.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'nmr_diffusion_analysis_{timestamp}.zip'
        )

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
        # Each upload gets its own UUID-namespaced directory to prevent
        # filename collisions when multiple users upload simultaneously.
        data_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        session_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], data_id)
        os.makedirs(session_upload_dir, exist_ok=True)
        filepath = os.path.join(session_upload_dir, filename)
        file.save(filepath)

        # Extract zip
        extract_dir = os.path.join(session_upload_dir, os.path.splitext(filename)[0])
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
            # Store a slim entry: omit browser-only keys and use float32 numpy for spectra
            _BROWSER_ONLY_KEYS = ('stacked_data', 'stacked_layout', 'selection_data')
            store_entry = {k: v for k, v in plot_data.items() if k not in _BROWSER_ONLY_KEYS}
            store_entry['raw_spectra'] = np.array(plot_data['raw_spectra'], dtype=np.float32)
            _nmr_data_store[data_id] = store_entry
            # Strip server-only keys (complex spectra) before sending to browser
            response_plot_data = {k: v for k, v in plot_data.items() if not k.startswith('_')}

            # Send first slice at full resolution for real-time client-side phase correction
            if '_complex_spectra' in plot_data and plot_data['_complex_spectra']:
                sp0 = plot_data['_complex_spectra'][0]
                ppm_arr = np.array(plot_data.get('raw_ppm', []))
                response_plot_data['complex_re_0'] = [float(v) for v in np.real(sp0)]
                response_plot_data['complex_im_0'] = [float(v) for v in np.imag(sp0)]
                response_plot_data['complex_ppm'] = ppm_arr.tolist()

            return jsonify({'message': 'File successfully uploaded and extracted', 'plot_data': response_plot_data, 'data_id': data_id})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Failed to process NMR data: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type. Please upload a .zip file'}), 400


def _apply_phase(complex_arr, ph0_deg, ph1_deg, ppm, pivot_ppm=None):
    """Apply 0th and 1st order phase correction to a (n_slices, n_pts) complex array."""
    n_pts = complex_arr.shape[1]
    ph0 = np.deg2rad(ph0_deg)
    ph1 = np.deg2rad(ph1_deg)
    pivot_idx = int(np.argmin(np.abs(ppm - float(pivot_ppm)))) if pivot_ppm is not None else n_pts // 2
    freq_axis = (np.arange(n_pts) - pivot_idx) / n_pts
    phase_vec = ph0 + ph1 * freq_axis
    return np.real(complex_arr * np.exp(1j * phase_vec[np.newaxis, :]))


@app.route('/phase_spectrum', methods=['POST'])
def phase_spectrum():
    """Return phase-corrected real spectra for live slider preview."""
    try:
        data = request.json
        data_id   = data.get('data_id')
        ph0       = float(data.get('ph0', 0.0))
        ph1       = float(data.get('ph1', 0.0))
        pivot_ppm = data.get('pivot_ppm')
        lb        = float(data.get('lb', 1.0))  # apodization parameter
        fft_points = data.get('fft_points')
        if fft_points:
            fft_points = int(fft_points)

        plot_data = _nmr_data_store.get(data_id)
        if not plot_data or '_complex_spectra' not in plot_data:
            return jsonify({'error': 'Data not found or no complex spectra stored'}), 404

        # Re-compute complex spectra if apodization or FFT points have changed
        recomputed_from_fid = (lb != 1.0 or fft_points) and '_raw_fid_slices' in plot_data
        if recomputed_from_fid:
            raw_fid_slices = plot_data['_raw_fid_slices']
            n_collected = plot_data.get('_n_collected', len(raw_fid_slices[0]) if raw_fid_slices else 0)
            n_fft_default = n_collected  # 1x default
            n_fft = fft_points if fft_points else n_fft_default
            
            # Re-process FID slices with new parameters
            complex_arr = []
            for trace in raw_fid_slices:
                # Apply exponential apodization with the provided lb value
                window = np.exp(-lb * np.pi * np.arange(len(trace)) / len(trace))
                trace_win = trace * window
                # FFT with variable zero-filling
                sp = np.fft.fftshift(np.fft.fft(trace_win, n_fft))
                complex_arr.append(sp)
            complex_arr = np.array(complex_arr)
        else:
            complex_arr = np.array(plot_data['_complex_spectra'])

        ppm = np.array(plot_data['raw_ppm'])
        # When n_fft differs from the stored spectrum length, interpolate ppm to match
        if complex_arr.shape[1] != len(ppm):
            ppm = np.linspace(ppm[0], ppm[-1], complex_arr.shape[1])

        phased = _apply_phase(complex_arr, ph0, ph1, ppm,
                              float(pivot_ppm) if pivot_ppm is not None else None)
        norm = float(np.max(np.abs(phased[0]))) or 1.0
        phased_norm = phased / norm

        resp = {
            'phased_spectra': [safe_list(sp) for sp in phased_norm],
            'ppm': ppm.tolist()
        }
        # Return raw complex so the client can switch back to instant client-side phase
        if recomputed_from_fid:
            resp['complex_re_0'] = [float(v) for v in np.real(complex_arr[0])]
            resp['complex_im_0'] = [float(v) for v in np.imag(complex_arr[0])]
        return jsonify(resp)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/auto_phase', methods=['POST'])
def auto_phase():
    """Find optimal ph0 + ph1 via negative-area minimisation (grid search + 2D Nelder-Mead)."""
    try:
        data = request.json
        data_id = data.get('data_id')

        plot_data = _nmr_data_store.get(data_id)
        if not plot_data or '_complex_spectra' not in plot_data:
            return jsonify({'error': 'Data not found'}), 404

        sp0 = np.array(plot_data['_complex_spectra'][0])
        n_pts = len(sp0)

        # Use the dominant peak as the pivot for ph1 (so ph1 doesn't shift ph0 at the main peak)
        mag = np.abs(sp0)
        pivot_idx = int(np.argmax(mag))

        freq_axis = (np.arange(n_pts) - pivot_idx) / n_pts  # normalized, zero at pivot

        def neg_area(ph0_deg, ph1_deg=0.0):
            phase_vec = np.deg2rad(ph0_deg) + np.deg2rad(ph1_deg) * freq_axis
            phased = np.real(sp0 * np.exp(1j * phase_vec))
            return float(np.sum(np.maximum(-phased, 0.0)))

        def neg_area_2d(params):
            return neg_area(params[0], params[1])

        # Step 1a: vectorized coarse grid (5° steps) on downsampled spectrum
        downsample = max(1, n_pts // 4096)
        sp0_ds = sp0[::downsample]
        fa_ds   = freq_axis[::downsample]
        coarse_grid = np.arange(-180.0, 180.0, 5.0)          # 72 angles
        ph0_rad = np.deg2rad(coarse_grid)[:, np.newaxis]       # (72,1)
        phased_grid = np.real(sp0_ds[np.newaxis, :] * np.exp(1j * ph0_rad))
        coarse_scores = np.sum(np.maximum(-phased_grid, 0.0), axis=1)
        best_coarse = float(coarse_grid[np.argmin(coarse_scores)])

        # Step 1b: fine serial search in ±10° window on full spectrum (1° steps)
        fine_grid = np.arange(best_coarse - 10.0, best_coarse + 11.0, 1.0)
        fine_scores = np.array([neg_area(p, 0.0) for p in fine_grid])
        best_ph0 = float(fine_grid[np.argmin(fine_scores)])

        # Step 2: 2D Nelder-Mead refines both ph0 and ph1 jointly
        result = minimize(neg_area_2d, [best_ph0, 0.0], method='Nelder-Mead',
                          options={'xatol': 0.1, 'fatol': 1.0, 'maxiter': 600,
                                   'adaptive': True})
        ph0_best, ph1_best = result.x
        # Wrap ph0 to [-180, 180]
        ph0_best = ((ph0_best + 180.0) % 360.0) - 180.0
        # Clamp ph1 to slider range
        ph1_best = float(np.clip(ph1_best, -1080.0, 1080.0))

        # Pivot ppm so the browser slider pivot matches what we used
        ppm_arr = np.array(plot_data.get('raw_ppm', []))
        pivot_ppm = float(ppm_arr[pivot_idx]) if len(ppm_arr) == n_pts else None

        return jsonify({'ph0': round(ph0_best, 2), 'ph1': round(ph1_best, 2),
                        'pivot_ppm': round(pivot_ppm, 4) if pivot_ppm is not None else None})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/apply_processing', methods=['POST'])
def apply_processing():
    """Apply phase correction + optional polynomial baseline, update stored spectra."""
    try:
        data = request.json
        data_id        = data.get('data_id')
        ph0            = float(data.get('ph0', 0.0))
        ph1            = float(data.get('ph1', 0.0))
        pivot_ppm      = data.get('pivot_ppm')
        baseline_order = int(data.get('baseline_order', -1))
        preview        = bool(data.get('preview', False))
        lb             = float(data.get('lb', 1.0))  # apodization parameter
        fft_points     = data.get('fft_points')
        if fft_points:
            fft_points = int(fft_points)

        plot_data = _nmr_data_store.get(data_id)
        if not plot_data or '_complex_spectra' not in plot_data:
            return jsonify({'error': 'Data not found or complex spectra unavailable'}), 404

        # Re-compute complex spectra if apodization or FFT points have changed
        if (lb != 1.0 or fft_points) and '_raw_fid_slices' in plot_data:
            raw_fid_slices = plot_data['_raw_fid_slices']
            n_collected = plot_data.get('_n_collected', len(raw_fid_slices[0]) if raw_fid_slices else 0)
            n_fft_default = n_collected  # 1x default
            n_fft = fft_points if fft_points else n_fft_default
            
            # Re-process FID slices with new parameters
            complex_spectra = []
            for trace in raw_fid_slices:
                # Apply exponential apodization with the provided lb value
                window = np.exp(-lb * np.pi * np.arange(len(trace)) / len(trace))
                trace_win = trace * window
                # FFT with variable zero-filling
                sp = np.fft.fftshift(np.fft.fft(trace_win, n_fft))
                complex_spectra.append(sp)
            
            # Update stored complex spectra
            plot_data['_complex_spectra'] = complex_spectra

        complex_arr = np.array(plot_data['_complex_spectra'])
        ppm = np.array(plot_data['raw_ppm'])
        # When n_fft differs from the stored spectrum length, interpolate ppm to match
        if complex_arr.shape[1] != len(ppm):
            ppm = np.linspace(ppm[0], ppm[-1], complex_arr.shape[1])

        # For preview, only process slice 0 (much faster)
        if preview:
            complex_arr = complex_arr[:1]

        # Phase correction
        phased = _apply_phase(complex_arr, ph0, ph1, ppm,
                              float(pivot_ppm) if pivot_ppm is not None else None)

        # Polynomial baseline correction
        baseline_0 = None  # baseline curve for first slice (preview overlay)
        if baseline_order >= 0:
            corrected = np.zeros_like(phased)
            n_pts = phased.shape[1]
            pts_idx = np.arange(n_pts)
            for i, sp in enumerate(phased):
                abs_sp = np.abs(sp)
                threshold = float(np.max(abs_sp)) * 0.05
                edge_width = max(int(n_pts * 0.1), baseline_order + 2)
                edge_mask = (pts_idx < edge_width) | (pts_idx >= n_pts - edge_width)
                low_signal_mask = abs_sp < threshold
                mask = edge_mask | low_signal_mask

                if np.sum(mask) > baseline_order + 1:
                    coeffs = np.polyfit(pts_idx[mask], sp[mask], baseline_order)
                    bl = np.polyval(coeffs, pts_idx)
                    candidate = sp - bl

                    original_max = float(np.max(np.abs(sp)))
                    candidate_max = float(np.max(np.abs(candidate)))
                    if original_max > 0 and candidate_max < original_max * 0.2:
                        corrected[i] = sp
                        if i == 0:
                            baseline_0 = None
                    else:
                        corrected[i] = candidate
                        if i == 0:
                            baseline_0 = bl
                else:
                    corrected[i] = sp
        else:
            corrected = phased

        # Normalize to positive max of first slice.
        # If the dominant (largest-magnitude) feature in slice 0 is negative
        # (e.g. Varian data with default ph0=0), flip all slices so peaks are positive.
        abs_max_val = float(np.max(np.abs(corrected[0])))
        if abs_max_val == 0:
            abs_max_val = 1.0
        peak_idx_c = int(np.argmax(np.abs(corrected[0])))
        if float(corrected[0, peak_idx_c]) < 0:
            corrected = -corrected
        max_v = float(np.max(corrected[0]))
        if max_v <= 0:
            max_v = abs_max_val
        corrected_norm = corrected / max_v

        processed_list = [safe_list(sp) for sp in corrected_norm]

        if not preview:
            # Permanently update stored spectra for subsequent analysis (float32 numpy)
            plot_data['raw_spectra'] = corrected_norm.astype(np.float32)
            # Keep ppm in sync with actual spectrum length (e.g. after non-1x FFT zero-fill)
            plot_data['raw_ppm'] = ppm.tolist()

        response = {'raw_spectra': processed_list, 'ppm': ppm.tolist()}
        if preview and baseline_0 is not None:
            # Normalize baseline to the same scale as applyPhaseJS (max(abs(phased[0])))
            phased_max = float(np.max(np.abs(phased[0]))) or 1.0
            response['baseline_0'] = safe_list(baseline_0 / phased_max)
        return jsonify(response)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def parse_varian_procpar(procpar_path):
    """
    Parse a Varian procpar file into a plain dict: {name: list_of_values}.
    Used as a fallback when nmrglue does not populate dic['procpar'].
    Each procpar parameter block has three parts:
      Line 1: name subtype basictype maxval minval step Ggroup Dgroup protection
      Line 2+: count val1 val2 ... valN  (may span multiple physical lines)
      Line 3: enum_count  [optional enum values]
    """
    result = {}
    try:
        with open(procpar_path, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            # --- descriptor line ---
            parts = lines[i].strip().split()
            if len(parts) < 3:
                i += 1
                continue
            name = parts[0]
            try:
                basictype = int(parts[2])   # 1 = real, 2 = string
            except (ValueError, IndexError):
                basictype = 1
            i += 1
            if i >= len(lines):
                break
            # --- values line(s) ---
            val_parts = lines[i].strip().split()
            i += 1
            if not val_parts:
                # skip enum line if present
                if i < len(lines):
                    i += 1
                continue
            try:
                count = int(val_parts[0])
                collected = list(val_parts[1:])
                # read continuation lines until we have 'count' values
                while len(collected) < count and i < len(lines):
                    extra = lines[i].strip()
                    i += 1
                    collected.extend(extra.split())
                raw = collected[:count]
                if basictype == 2:
                    result[name] = [v.strip('"\'') for v in raw]
                else:
                    converted = []
                    for v in raw:
                        try:
                            converted.append(float(v))
                        except ValueError:
                            converted.append(v)
                    result[name] = converted
            except (ValueError, IndexError):
                result[name] = []
            # --- enum count line (always present, usually "0") ---
            if i < len(lines):
                enum_line = lines[i].strip().split()
                if enum_line:
                    try:
                        enum_count = int(enum_line[0])
                        i += 1
                        # skip enum_count value lines (each on own line, or all on one)
                        for _ in range(enum_count):
                            if i < len(lines):
                                i += 1
                    except ValueError:
                        pass  # not an enum line; leave i unchanged
    except Exception as e:
        print(f'procpar parse error: {e}')
    return result


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

def detect_sequence_type(pulse_program):
    """
    Detect the diffusion pulse sequence type from pulse program name.
    
    PGSE variants (Pulsed Gradient Spin Echo):
    - stebpgp1s, bipgp*, pgse, ged, zg_pulsed_eff, ledbpgp2s
    Equation: ST_X = (γ·g·δ)² · (Δ - δ/3)
    
    PGSTE variants (Pulsed Gradient Stimulated Echo):
    - sted, bppgste, pgste, stimulated
    Equation: ST_X = (γ·g·δ)² · (Δ - δ/3 - δ₂/2)  [requires storage pulse duration δ₂]
    Note: Bruker typically uses Δ - δ/3 for PGSTE as well in practice
    """
    pulse_program = str(pulse_program).lower().strip()
    
    # Stimulated echo variants — includes bipolar PGSTE (dbppste, bppste, dbppste_cc)
    if any(x in pulse_program for x in ['sted', 'pgste', 'stimulated', 'ppste', 'bppste']):
        return 'PGSTE'
    elif any(x in pulse_program for x in ['stebpgp', 'bipgp', 'pgse', 'ged', 'led', 'bppgp']):
        return 'PGSE'
    else:
        return 'PGSE'  # Default to PGSE for unknown sequences

def detect_gradient_shape(pulse_program, dic=None, params=None):
    """
    Detect gradient shape from Bruker GPNAM parameter (when available), falling
    back to pulse program name heuristics.

    shape_factor is applied as: g_eff = g_nominal * shape_factor
    This corrects for the reduced effective gradient area of shaped pulses.

    Shape factors (first-moment / area relative to rectangular pulse):
      square / SMSQ* : ~1.0  (Smoothed Square is effectively rectangular)
      sinusoidal SINE: 2/π ≈ 0.6366
      gaussian       : ~0.7  (approximate)
    """
    TWO_OVER_PI = 2.0 / math.pi  # ≈ 0.6366 — true sinusoidal correction

    # ── 1. Read actual GPNAM from Bruker dic ─────────────────────────────────
    if dic is not None:
        acqus = dic.get('acqus', {})
        gpnam_raw = acqus.get('GPNAM', [])

        # nmrglue parses GPNAM as a list of strings (brackets stripped)
        if isinstance(gpnam_raw, (list, tuple)):
            shapes = [str(s).strip() for s in gpnam_raw
                      if str(s).strip() and str(s).strip() not in ('', '<>')]
        else:
            # Fallback: raw string — parse <name> tokens manually
            import re as _re
            shapes = [m for m in _re.findall(r'<([^>]+)>', str(gpnam_raw)) if m.strip()]

        if shapes:
            # Use the most common non-empty shape name
            from collections import Counter
            most_common = Counter(shapes).most_common(1)[0][0].lower()
            if most_common.startswith('smsq') or 'smooth' in most_common:
                # Smoothed-square (e.g. SMSQ10.100): effectively rectangular
                return 'square', 1.0
            elif most_common.startswith('sine') or most_common.startswith('sin.'):
                return 'sinusoidal', TWO_OVER_PI
            elif 'gauss' in most_common:
                return 'gaussian', 0.70
            else:
                # Unknown shape — treat as square (conservative)
                return 'square', 1.0

    # ── 2. Fallback: infer from pulse program name ────────────────────────────
    pulse_program = str(pulse_program).lower().strip()

    if any(x in pulse_program for x in ['square', 'sqr', 'hard', 'rect']):
        return 'square', 1.0
    if any(x in pulse_program for x in ['sine', 'smooth', 'shaped']):
        return 'sinusoidal', TWO_OVER_PI

    # Common Bruker DOSY sequences default to SMSQ gradients → treat as square
    if any(x in pulse_program for x in ['stebpgp', 'bipgp', 'ledbpgp', 'dstegp', 'sted']):
        return 'square', 1.0

    return 'square', 1.0

def process_nmr_data(extract_dir, lb=1.0, fft_points=None):
    vendor, data_path = find_bruker_or_varian(extract_dir)

    if not vendor:
        raise ValueError("Could not detect Bruker or Varian data structure in the uploaded zip.")

    params = {'gamma': 2.67522e8, 'vendor': vendor}
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
                # P[30] is the gradient half-lobe duration in stebpgp1s/stebpgp1s19;
                # total delta = 2 * P[30].  P[40] (if populated) often stores the
                # full gradient duration directly (= 2*P[30]) and serves as fallback
                # for sequences that don't use P[30].
                val_p30 = get_bruker_item(dic, acqus_path, 'P', 30)
                if val_p30 is not None and val_p30 > 0:
                    # Little delta (total gradient duration) = 2 * P[30]
                    params['delta'] = 2.0 * val_p30 / 1000000.0
                else:
                    # Fallback: some sequences store the full gradient pulse
                    # duration in P[40] (= 2*P[30] in stebpgp1s-family).
                    val_p40 = get_bruker_item(dic, acqus_path, 'P', 40)
                    if val_p40 is not None and val_p40 > 0:
                        params['delta'] = val_p40 / 1000000.0
                
                # D[20] is the diffusion time (Big Delta)
                val_d20 = get_bruker_item(dic, acqus_path, 'D', 20)
                if val_d20 is not None:
                    params['big_delta'] = val_d20

                # D[16] is the inter-bipolar-gradient delay τ used in
                # Bruker bipolar PGSTE sequences (stebpgp1s, stebpgp1s19, etc.)
                val_d16 = get_bruker_item(dic, acqus_path, 'D', 16)
                if val_d16 is not None:
                    params['tau_bipolar'] = val_d16

                # ── Additional spectrometer/acquisition parameters ──
                acqus_d = dic.get('acqus', {})
                try:
                    sfo1 = float(acqus_d.get('SFO1', 0) or 0)
                    if sfo1 > 0:
                        params['sfo1_mhz'] = sfo1  # exact MHz (e.g. 400.1324)
                        sw_p = float(acqus_d.get('SW', 0) or 0)
                        if sw_p > 0:
                            params['sw_ppm'] = sw_p
                            params['sw_hz'] = sw_p * sfo1
                except Exception:
                    pass
                try:
                    ns = int(acqus_d.get('NS', 0) or 0)
                    if ns > 0:
                        params['n_scans'] = ns
                except Exception:
                    pass
                val_d1 = get_bruker_item(dic, acqus_path, 'D', 1)
                if val_d1 is not None and val_d1 > 0:
                    params['recycle_delay'] = val_d1

                print(f"[Bruker param extraction] delta={params.get('delta')} big_delta={params.get('big_delta')} tau={params.get('tau_bipolar')} P30={val_p30} D20={val_d20} D16={val_d16}")
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

        # nmrglue may return (1, n_gradients, n_pts) or higher-dimensional arrays
        # for arrayed Varian experiments. Squeeze singleton dims then ensure 2D.
        data = np.squeeze(data)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        elif data.ndim > 2:
            data = data.reshape(-1, data.shape[-1])

        procpar = dic.get('procpar', {})

        # Helpers that work for both nmrglue format {'values': [...]} and our manual parser format [...]
        def get_pp(name, default=None, idx=0, as_str=False):
            entry = procpar.get(name)
            if entry is None:
                return default
            vals = entry.get('values', []) if isinstance(entry, dict) else entry
            if not vals or idx >= len(vals):
                return default
            try:
                v = vals[idx]
                return str(v).strip('"\'') if as_str else float(v)
            except (ValueError, TypeError):
                return str(vals[idx]).strip('"\'') if as_str else default

        def get_pp_array(name):
            entry = procpar.get(name)
            if entry is None:
                return []
            vals = entry.get('values', []) if isinstance(entry, dict) else entry
            out = []
            for v in vals:
                try:
                    out.append(float(v))
                except (ValueError, TypeError):
                    pass
            return out

        # If nmrglue procpar is empty, fall back to manual parser
        if not procpar:
            procpar_path_v = os.path.join(data_path, 'procpar')
            if os.path.exists(procpar_path_v):
                procpar = parse_varian_procpar(procpar_path_v)
                print(f"Varian: manual procpar parser found {len(procpar)} params")

        def first_pp_value(names, as_str=False):
            for name in names:
                value = get_pp(name, default=None, as_str=as_str)
                if value is not None:
                    return value
            return None

        # --- Big delta (diffusion delay) ---
        # Varian DOSY sequences commonly use procpar names such as del/Ddelta/del2.
        big_delta_v = first_pp_value(['del', 'Ddelta', 'del2'])
        if big_delta_v is not None:
            params['big_delta'] = big_delta_v

        # --- Little delta (gradient pulse duration) ---
        # Dbppste/Dbppste_cc commonly uses gt1; other variants may store dro/delta/del1.
        delta_v = first_pp_value(['gt1', 'dro', 'delta', 'del1'])
        if delta_v is not None:
            params['delta'] = delta_v

        # --- Gradient levels array ---
        gzlvl1 = get_pp_array('gzlvl1')
        if not gzlvl1:
            gzlvl1 = get_pp_array('gzlvlw')   # some sequences use gzlvlw

        if gzlvl1:
            abs_gz = [abs(g) for g in gzlvl1]
            max_gz = max(abs_gz) if abs_gz else 1.0
            # Keep raw DAC units — do NOT normalize to 0-1 fractions.
            # The calibration step fits G/cm per DAC unit (gcal) directly.
            difframp = abs_gz
            difflist = abs_gz
            params['dac_max'] = max_gz  # highest DAC value used

            # Gradient calibration factor (G/cm per DAC unit) if available
            gcal = get_pp('gcal_', default=None)
            if gcal is None:
                gcal = get_pp('gcal', default=None)
            if gcal is not None:
                params['gcal'] = gcal
                params['gmax_gcm'] = max_gz * gcal

        # --- Pulse sequence name ---
        seqfil = get_pp('seqfil', default='unknown', as_str=True)
        pulse_program = seqfil.lower().strip('"\'')
        params['pulse_program'] = pulse_program
        
        # Detect sequence type and gradient shape for analysis
        sequence_type = detect_sequence_type(pulse_program)
        gradient_shape, gradient_shape_factor = detect_gradient_shape(pulse_program, params=params)
        params['sequence_type'] = sequence_type
        params['gradient_shape'] = gradient_shape
        params['gradient_shape_factor'] = gradient_shape_factor

        # --- Bipolar inter-lobe delay (tau_bipolar / τ) ---
        # For Varian bipolar PGSTE (Dbppste, bppste, dbppste_cc):
        #   gstab  = stabilisation delay between the two lobes of each bipolar pair
        #   This is τ in the equation: b = (γgδ)²·(Δ − δ/3 − τ/2)
        if sequence_type == 'PGSTE':
            tau_v = first_pp_value(['gstab', 'tDELTA', 'tau1', 'tau'])
            params['tau_bipolar'] = float(tau_v) if tau_v is not None else 0.0
        else:
            params['tau_bipolar'] = 0.0

        # --- Spectrometer frequency, spectral width, and transmitter offset ---
        sfrq_v = get_pp('sfrq', default=None)   # MHz
        sw_v   = get_pp('sw',   default=None)   # Hz
        tof_v  = get_pp('tof',  default=0.0)    # Hz — transmitter offset from carrier
        if sfrq_v:
            params['sfrq'] = sfrq_v
            params['sfo1_mhz'] = sfrq_v
        if sw_v:
            params['sw'] = sw_v
            params['sw_hz'] = sw_v
            if sfrq_v:
                params['sw_ppm'] = sw_v / sfrq_v
        if tof_v:
            params['tof'] = tof_v

        # nt — number of transients per gradient step
        nt_v = get_pp('nt', default=None)
        if nt_v is not None:
            try:
                params['n_scans'] = int(nt_v)
            except Exception:
                pass

        # d1 — recycle delay (s)
        d1_v = get_pp('d1', default=None)
        if d1_v is not None:
            params['recycle_delay'] = d1_v

        # Align number of FID slices with gradient array length
        if gzlvl1 and len(data.shape) == 2 and data.shape[0] != len(gzlvl1):
            n_match = min(data.shape[0], len(gzlvl1))
            data = data[:n_match]
            difframp = difframp[:n_match]
            difflist = difflist[:n_match]

        # Ensure data is complex — Varian fid interleaves real/imag as 32-bit int pairs
        if data.dtype.kind in ('i', 'u'):
            data = data.astype(np.float64)
        if not np.iscomplexobj(data):
            if len(data.shape) == 2 and data.shape[1] % 2 == 0:
                data = data[:, ::2] + 1j * data[:, 1::2]
            elif len(data.shape) == 1 and len(data) % 2 == 0:
                data = data[::2] + 1j * data[1::2]

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
        
        # Detect sequence type and gradient shape for analysis
        sequence_type = detect_sequence_type(pulse_program)
        gradient_shape, gradient_shape_factor = detect_gradient_shape(pulse_program, dic=dic, params=params)
        params['sequence_type'] = sequence_type
        params['gradient_shape'] = gradient_shape
        params['gradient_shape_factor'] = gradient_shape_factor

        if 'd2o' in folder_search or 'd2o' in pulse_program:
            detected_standard = 'D2O'
        elif 'glycerol' in folder_search or 'glyc' in pulse_program:
            detected_standard = 'Glycerol'
        elif 'squalane' in folder_search or 'squal' in pulse_program:
            detected_standard = 'Squalane'
    elif vendor == 'varian':
        folder_parts = os.path.normpath(data_path).split(os.sep)
        folder_search = ' '.join(folder_parts).lower()
        pulse_prog_v  = params.get('pulse_program', '')
        if 'd2o' in folder_search or 'd2o' in pulse_prog_v:
            detected_standard = 'D2O'
        elif 'glycerol' in folder_search or 'glyc' in folder_search:
            detected_standard = 'Glycerol'
        elif 'squalane' in folder_search or 'squal' in folder_search:
            detected_standard = 'Squalane'

    # Calculate magnitude spectra for all slices
    processed_spectra = []  # magnitude for stacked display
    complex_spectra = []    # complex for server-side phase correction
    
    # Determine FFT size: use provided fft_points or default to 1x (no zero-fill)
    n_collected = len(slices[0]) if slices else 0
    params['n_collected'] = n_collected  # expose to browser for FFT picker
    n_fft_default = n_collected  # 1x by default — fast, real-time sliders
    n_fft = fft_points if fft_points else n_fft_default
    
    for i, trace in enumerate(slices):
        # 1. Exponential apodization with user-specified line broadening (default lb=1 Hz)
        window = np.exp(-lb * np.pi * np.arange(len(trace)) / len(trace))
        trace_win = trace * window

        # 2. FT with variable zero-filling
        sp = np.fft.fftshift(np.fft.fft(trace_win, n_fft))
        processed_spectra.append(np.abs(sp))
        complex_spectra.append(sp)  # retain complex for phase correction

    # Normalize to the maximum of the FIRST gradient slice
    norm_factor = np.max(processed_spectra[0]) if len(processed_spectra) > 0 and np.max(processed_spectra[0]) > 0 else 1.0

    # Compute ppm axis from Bruker acqus parameters (SW, SFO1, O1)
    ppm_base = None
    if vendor == 'bruker':
        try:
            sw_ppm = float(dic['acqus']['SW'])
            sf     = float(dic['acqus']['SFO1'])
            o1     = float(dic['acqus']['O1'])
            o1p    = o1 / sf  # Hz / MHz = ppm
            n_pts  = len(complex_spectra[0])
            ppm_base = np.linspace(o1p + sw_ppm / 2, o1p - sw_ppm / 2, n_pts)
        except Exception as ppm_err:
            print(f'PPM axis from acqus failed: {ppm_err}')
    elif vendor == 'varian':
        try:
            sfrq_v = params.get('sfrq')   # MHz
            sw_v   = params.get('sw')     # Hz
            tof_v  = params.get('tof', 0.0)   # Hz
            if sfrq_v and sw_v:
                sw_ppm_v  = sw_v  / sfrq_v   # Hz / MHz = ppm
                ctr_ppm_v = tof_v / sfrq_v   # Hz / MHz = ppm
                n_pts_v   = len(complex_spectra[0])
                ppm_base  = np.linspace(ctr_ppm_v + sw_ppm_v / 2,
                                        ctr_ppm_v - sw_ppm_v / 2, n_pts_v)
        except Exception as ppm_err_v:
            print(f'Varian PPM axis failed: {ppm_err_v}')
    if ppm_base is None:
        ppm_base = np.linspace(15, -5, len(complex_spectra[0]))

    traces = []
    for i, sp_mag in enumerate(processed_spectra):
        sp_norm = sp_mag / norm_factor
        
        # Vertical offset for stacking
        y_offset = i * 0.05
        
        # Horizontal shift (ppm) to create a staggered waterfall effect
        x_shift = i * 0.1
        x_ppm_staggered = (ppm_base - x_shift)

        # Labels for the plot
        if params.get('vendor') == 'varian':
            # Varian: show raw DAC units, not percentages
            label_str = f'DAC: {int(difframp[i])}' if difframp is not None else f'{i+1}'
        else:
            # Bruker: difframp is 0-1 fraction → show as %
            label_val = (difframp[i] * 100) if difframp is not None else difflist[i]
            label_str = f'{label_val:.2f}%'

        traces.append({
            'x': x_ppm_staggered.tolist(),
            'y': (sp_norm + y_offset).tolist(),
            'customdata': ppm_base.tolist(), # Store original ppm here
            'type': 'scatter',
            'mode': 'lines',
            'name': f'Step {i+1} ({label_str})',
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
        'detected_standard': detected_standard,
        'initial_processing_lb': lb,
        'initial_processing_fft_points': fft_points,
        'processing_lb': lb,
        'processing_fft_points': fft_points,
        '_complex_spectra': complex_spectra,  # numpy arrays; stripped before browser response
        '_raw_fid_slices': slices,  # raw FID data; kept server-side for re-processing
        '_n_collected': n_collected  # number of collected FID points
    }

@app.route('/get_calibrations', methods=['GET'])
def get_calibrations():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, timestamp, max_g_gauss, delta, big_delta, fit_slope, fit_intercept, vendor FROM calibrations ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{
        'id': r[0], 'name': r[1], 'timestamp': r[2],
        'max_g': r[3], 'delta': r[4], 'big_delta': r[5],
        'slope': r[6], 'intercept': r[7],
        'vendor': r[8] or 'bruker'
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
    DELTA = exp_params.get('delta')
    BIG_DELTA = exp_params.get('big_delta')

    missing_params = []
    if DELTA is None:
        missing_params.append('delta')
    if BIG_DELTA is None:
        missing_params.append('big_delta')
    if missing_params:
        return jsonify({
            'error': (
                'Dataset is missing required diffusion timing parameter(s): '
                + ', '.join(missing_params)
                + '. For Bruker data, delta is read from P[30] (or P[40]) in acqus; '
                  'big_delta from D[20]. Re-upload or check that acqus contains these values.'
            )
        }), 400
    
    # If no diff_ramp provided, default to linear 0.02 to 0.95
    if len(diff_ramp) == 0:
        diff_ramp = np.linspace(0.02, 0.95, len(intensities))
    
    vendor = exp_params.get('vendor', 'bruker')
    TAU_BIPOLAR = float(exp_params.get('tau_bipolar', 0.0) or 0.0)
    GRADIENT_SHAPE_FACTOR = float(exp_params.get('gradient_shape_factor', 1.0) or 1.0)
    SEQUENCE_TYPE = exp_params.get('sequence_type', 'PGSE')
    GRADIENT_SHAPE = exp_params.get('gradient_shape', 'square')

    if vendor == 'varian':
        # Varian: diff_ramp contains raw DAC units (e.g. 100 … 32000).
        # Fit gcal = G/cm per DAC unit.  g(T/m) = dac * gcal * 0.01
        def stejskal_tanner(dac_val, I0, gcal_gcm_per_dac):
            g_tm = dac_val * gcal_gcm_per_dac * 0.01 * GRADIENT_SHAPE_FACTOR
            b_value = (GAMMA * g_tm * DELTA)**2 * (BIG_DELTA - DELTA / 3.0 - TAU_BIPOLAR / 2.0)
            return I0 * np.exp(-D_known * b_value)

        # Initial guess: assume ~40 G/cm max at the highest DAC value
        dac_max_guess = float(np.max(diff_ramp)) if len(diff_ramp) > 0 else 32767.0
        gcal_guess = 40.0 / dac_max_guess  # G/cm per DAC unit
        p0_cal = [intensities[0], gcal_guess]
        bounds_cal = ([0, 0], [np.inf, 10.0])
    else:
        # Bruker: diff_ramp is 0-1 fraction; fit g_scale = max gradient in T/m
        def stejskal_tanner(ramp_val, I0, g_scale):
            g = ramp_val * g_scale * GRADIENT_SHAPE_FACTOR
            b_value = (GAMMA * g * DELTA)**2 * (BIG_DELTA - DELTA / 3.0 - TAU_BIPOLAR / 2.0)
            return I0 * np.exp(-D_known * b_value)

        p0_cal = [intensities[0], 0.5]
        bounds_cal = ([0, 0], [np.inf, np.inf])

    try:
        popt, _ = curve_fit(stejskal_tanner, diff_ramp, intensities,
                            p0=p0_cal, bounds=bounds_cal, maxfev=10000)

        I0_fit = popt[0]
        fit_intensities = stejskal_tanner(diff_ramp, *popt)

        # Dense smooth fit line (500 points across the DAC / ramp range)
        ramp_smooth = np.linspace(0.0, float(np.max(diff_ramp)), 500)
        fit_line_y  = stejskal_tanner(ramp_smooth, *popt)

        if vendor == 'varian':
            gcal_fit       = popt[1]              # G/cm per DAC unit
            gradients_gauss = np.array(diff_ramp) * gcal_fit   # G/cm
            calculated_max_g_gauss = float(np.max(diff_ramp)) * gcal_fit
            fit_line_x = (ramp_smooth * gcal_fit).tolist()    # G/cm axis
        else:
            G_max_fit = popt[1]                   # T/m
            calculated_max_g_gauss = G_max_fit * 100.0        # G/cm
            gradients_gauss = diff_ramp * calculated_max_g_gauss
            fit_line_x = (ramp_smooth * calculated_max_g_gauss).tolist()

        # Linear calibration: G/cm = slope * x + intercept
        # x = DAC units (Varian) or 0-1 fraction (Bruker)
        slope, intercept = np.polyfit(diff_ramp, gradients_gauss, 1)

    except Exception as e:
        return jsonify({'error': f'Fitting failed: {str(e)}'}), 500

    return jsonify({
        'ppm': ppm_clicked,
        'intensities': intensities.tolist(),
        'fit_intensities': fit_intensities.tolist(),
        'fit_line': {
            'x': fit_line_x,
            'y': fit_line_y.tolist()
        },
        'gradient_steps': diff_ramp.tolist(),
        'gradients': gradients_gauss.tolist(),
        'calibrated_max_g': calculated_max_g_gauss,
        'fit_slope': float(slope),
        'fit_intercept': float(intercept),
        'delta': DELTA,
        'big_delta': BIG_DELTA,
        'tau_bipolar': TAU_BIPOLAR,
        'gradient_shape_factor': GRADIENT_SHAPE_FACTOR,
        'sequence_type': SEQUENCE_TYPE,
        'gradient_shape': GRADIENT_SHAPE,
        'difflist': data.get('difflist', []),
        'vendor': vendor
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
    vendor = data.get('vendor', 'bruker')

    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO calibrations (name, max_g_gauss, delta, big_delta, fit_slope, fit_intercept, vendor)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, max_g, delta, big_delta, slope, intercept, vendor))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Calibration saved successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── Session History Endpoints ───────────────────────────────────────────────

@app.route('/api/restore_data/<data_id>', methods=['GET'])
def restore_data(data_id):
    """Re-process uploaded NMR data from disk by data_id and return full plot_data."""
    uuid_re = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)
    if not uuid_re.match(data_id):
        return jsonify({'error': 'Invalid data_id'}), 400

    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], data_id)
    if not os.path.isdir(upload_dir):
        return jsonify({'error': 'Data not found — it may have been deleted after 3 days'}), 404

    # Find the extract subdirectory (first non-zip entry)
    extract_dir = None
    for entry in os.scandir(upload_dir):
        if entry.is_dir():
            extract_dir = entry.path
            break
    if not extract_dir:
        return jsonify({'error': 'Extracted data directory not found'}), 404

    try:
        plot_data = process_nmr_data(extract_dir)
        _BROWSER_ONLY_KEYS = ('stacked_data', 'stacked_layout', 'selection_data')
        store_entry = {k: v for k, v in plot_data.items() if k not in _BROWSER_ONLY_KEYS}
        store_entry['raw_spectra'] = np.array(plot_data['raw_spectra'], dtype=np.float32)
        _nmr_data_store[data_id] = store_entry
        response_plot_data = {k: v for k, v in plot_data.items() if not k.startswith('_')}
        if '_complex_spectra' in plot_data and plot_data['_complex_spectra']:
            sp0 = plot_data['_complex_spectra'][0]
            ppm_arr = np.array(plot_data.get('raw_ppm', []))
            response_plot_data['complex_re_0'] = [float(v) for v in np.real(sp0)]
            response_plot_data['complex_im_0'] = [float(v) for v in np.imag(sp0)]
            response_plot_data['complex_ppm'] = ppm_arr.tolist()
        return jsonify({'plot_data': response_plot_data, 'data_id': data_id})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/session/save_analysis', methods=['POST'])
def save_session_analysis():
    """Save an analysis result to the session history."""
    try:
        data = request.json
        session_id = data.get('session_id', '').strip()
        data_id = data.get('data_id', '').strip()
        if not session_id or not data_id:
            return jsonify({'error': 'session_id and data_id are required'}), 400
        uuid_re = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
        )
        if not uuid_re.match(session_id) or not uuid_re.match(data_id):
            return jsonify({'error': 'Invalid ID format'}), 400
        dataset_name = str(data.get('dataset_name', 'Unknown Dataset'))[:200]
        summary = data.get('summary', {})
        conn = sqlite3.connect(DB_FILE)
        conn.execute(
            """INSERT OR REPLACE INTO user_sessions
               (session_id, data_id, dataset_name, created_at, summary_json)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, data_id, dataset_name, time.time(), json.dumps(summary))
        )
        conn.commit()
        conn.close()
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/session/history/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """Return saved analyses for a session (most recent first)."""
    try:
        uuid_re = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
        )
        if not uuid_re.match(session_id):
            return jsonify([])
        conn = sqlite3.connect(DB_FILE)
        rows = conn.execute(
            """SELECT data_id, dataset_name, created_at, summary_json
               FROM user_sessions WHERE session_id = ?
               ORDER BY created_at DESC""",
            (session_id,)
        ).fetchall()
        conn.close()
        items = []
        for row in rows:
            try:
                summary = json.loads(row[3]) if row[3] else {}
            except Exception:
                summary = {}
            items.append({
                'data_id': row[0],
                'dataset_name': row[1],
                'created_at': row[2],
                'summary': summary
            })
        return jsonify(items)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/session/delete/<data_id>', methods=['DELETE'])
def delete_session_analysis(data_id):
    """Remove a saved analysis from the session history."""
    try:
        uuid_re = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I
        )
        if not uuid_re.match(data_id):
            return jsonify({'error': 'Invalid ID'}), 400
        conn = sqlite3.connect(DB_FILE)
        conn.execute("DELETE FROM user_sessions WHERE data_id = ?", (data_id,))
        conn.commit()
        conn.close()
        if data_id in _nmr_data_store:
            del _nmr_data_store[data_id]
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], data_id)
        if os.path.isdir(upload_dir):
            shutil.rmtree(upload_dir, ignore_errors=True)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _cleanup_old_sessions():
    """Delete session data older than 3 days. Reschedules itself every 6 h."""
    try:
        cutoff = time.time() - 3 * 86400
        conn = sqlite3.connect(DB_FILE)
        old_rows = conn.execute(
            "SELECT data_id FROM user_sessions WHERE created_at < ?", (cutoff,)
        ).fetchall()
        conn.execute("DELETE FROM user_sessions WHERE created_at < ?", (cutoff,))
        conn.commit()
        conn.close()
        for (did,) in old_rows:
            if did in _nmr_data_store:
                del _nmr_data_store[did]
            upload_dir = os.path.join('uploads', did)
            if os.path.isdir(upload_dir):
                shutil.rmtree(upload_dir, ignore_errors=True)
        if old_rows:
            print(f"[cleanup] Removed {len(old_rows)} expired session(s).")
    except Exception as e:
        print(f"[cleanup] Error: {e}")
    finally:
        t = threading.Timer(6 * 3600, _cleanup_old_sessions)
        t.daemon = True
        t.start()


# Start background cleanup (first run 60 s after startup)
_t = threading.Timer(60, _cleanup_old_sessions)
_t.daemon = True
_t.start()

if __name__ == '__main__':
    app.run(port=3000, debug=True, use_reloader=False)
