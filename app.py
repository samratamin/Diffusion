import os
import zipfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import nmrglue as ng
import plotly.graph_objects as go
import numpy as np
import json
import sqlite3

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

    if vendor == 'bruker':
        # Read the raw Bruker data
        dic, data = ng.bruker.read(data_path)
        
        # Determine the shape and format. 
        # Bruker pseudo-2D data can be (N_gradients, N_points) where N_points includes 
        # interleaved real/imaginary, or it can be (N_gradients, N_complex_points).
        # nmrglue's remove_digital_filter should be applied to the raw data once.
        data = ng.bruker.remove_digital_filter(dic, data)
        
        # If after digital filtering the data is not complex, and the first dimension
        # is even, we likely have interleaved Real/Imaginary FIDs.
        if not np.iscomplexobj(data) and len(data.shape) == 2:
            # Reconstruct complex: [R1, I1, R2, I2, ...] -> [R1+jI1, R2+jI2, ...]
            # This handles the 32-slice vs 16-slice issue correctly.
            data = data[::2] + 1j * data[1::2]
            
    elif vendor == 'varian':
        dic, data = ng.varian.read(data_path)

    # Convert to complex if still not complex
    if not np.iscomplexobj(data):
        data = data.astype(complex)

    # data should now be (N_gradients, N_points)
    if len(data.shape) == 1:
        slices = [data]
    else:
        slices = [data[i] for i in range(data.shape[0])]

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

    traces = []
    # Create a consistent ppm base scale
    ppm_base = np.linspace(15, -5, len(processed_spectra[0]))

    for i, sp_mag in enumerate(processed_spectra):
        sp_norm = sp_mag / norm_factor
        
        # Vertical offset for stacking
        y_offset = i * 0.05
        
        # Horizontal shift (ppm) to create a staggered waterfall effect
        # Shifting to the right (lower ppm) by 0.1 ppm per slice
        x_shift = i * 0.1
        x_ppm_staggered = (ppm_base - x_shift).tolist()

        traces.append({
            'x': x_ppm_staggered,
            'y': (sp_norm + y_offset).tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': f'Gradient {i+1}'
        })

    layout = {
        'title': f'NMR {vendor.capitalize()} Diffusion Data (Pseudo-2D Stacked Plot)',
        'xaxis': {'title': 'Chemical Shift (ppm)', 'autorange': 'reversed'},
        'yaxis': {'title': 'Normalized Intensity', 'showticklabels': False},
        'showlegend': True,
        'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
    }

    return {'data': traces, 'layout': layout}

if __name__ == '__main__':
    app.run(port=3000, debug=True)
