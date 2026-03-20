import os
import zipfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import nmrglue as ng
import plotly.graph_objects as go
import numpy as np
import json

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
            plot_data = process_nmr_data(extract_dir)
            return jsonify({'message': 'File successfully uploaded and extracted', 'plot_data': plot_data})
        except Exception as e:
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
        dic, data = ng.bruker.read(data_path)
        # Remove digital filter if necessary (depends on Bruker format, safe to try)
        if dic.get('acqus') and dic['acqus'].get('GRPDLY'):
            data = ng.bruker.remove_digital_filter(dic, data)
    elif vendor == 'varian':
        dic, data = ng.varian.read(data_path)

    # Process pseudo-2D
    # It might be 1D if it's not a diffusion set, but diffusion sets are pseudo-2D
    if len(data.shape) == 1:
        # Just a single 1D spec
        slices = [data]
    else:
        # Take the slices (gradients)
        slices = [data[i] for i in range(data.shape[0])]

    traces = []

    # Simple processing for visualization: FT and phase (auto-phase is hard, just FT for now)
    # Just show absolute value spectrum for basic visualization
    for i, trace in enumerate(slices):
        # basic FT
        sp = np.fft.fftshift(np.fft.fft(trace))
        # use magnitude spectrum for robust initial visualization
        sp_mag = np.abs(sp)

        # Real data tends to be huge, let's decimate or just plot as is if small enough
        x_axis = np.arange(len(sp_mag))

        # Offset each slice on the y-axis to create a stacked/waterfall effect
        offset = i * (np.max(sp_mag) * 0.1) if i > 0 and len(slices) > 1 else 0

        traces.append({
            'x': x_axis.tolist(),
            'y': (sp_mag + offset).tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': f'Gradient {i+1}'
        })

    layout = {
        'title': 'NMR Diffusion Data (Pseudo-2D Stacked Plot)',
        'xaxis': {'title': 'Points', 'autorange': 'reversed'},  # NMR data often plotted reversed
        'yaxis': {'title': 'Intensity (Offset)', 'showticklabels': False},
        'showlegend': True,
        'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
    }

    return {'data': traces, 'layout': layout}

if __name__ == '__main__':
    app.run(port=3000, debug=True)
