from flask import Flask, request, jsonify, render_template
import base64
from werkzeug.utils import secure_filename
import io
import os

from predictor import load_models, predict_image_from_bytes

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file

# Load models on startup
load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("=== PREDICT ENDPOINT CALLED ===")
    try:
        if 'image' in request.files:
            print("Processing file upload")
            # File upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            image_bytes = file.read()
        elif 'base64' in request.form:
            print("Processing base64 image (webcam)")
            # Webcam base64
            data_url = request.form['base64']
            image_bytes = base64.b64decode(data_url.split(',')[1])
        else:
            print("No image provided")
            return jsonify({'error': 'No image provided'}), 400
        
        print(f"Image bytes length: {len(image_bytes)}")
        result = predict_image_from_bytes(image_bytes)
        print(f"Prediction result: {result}")
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    print("Server starting at http://localhost:5000")
    print("Models loaded:", 'predictor.py' in os.listdir('.'))
    app.run(debug=True, host='0.0.0.0', port=5000)
