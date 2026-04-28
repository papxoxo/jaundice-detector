import cv2
import numpy as np
import joblib
import tempfile
import os

from utils import resize_image, gray_world_white_balance, segment_sclera, extract_features

MODEL_PATH = "models/svm_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None

def load_models():
    global model, scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

def predict_image_from_bytes(image_bytes):
    """Predict from image bytes (uploaded file or webcam base64)"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name
    
    try:
        img = cv2.imread(tmp_path)
        if img is None:
            return {'error': 'Invalid image'}
        
        img = resize_image(img)
        img = gray_world_white_balance(img)
        
        sclera, mask = segment_sclera(img)
        features = extract_features(img, mask)
        
        if features is None:
            return {'error': 'Sclera not detected'}
        
        features = scaler.transform([features])
        prediction = model.predict(features)[0]
        prob_jaundice = model.predict_proba(features)[0][1]
        
        result = 'jaundice' if prediction == 1 else 'normal'
        confidence = prob_jaundice if prediction == 1 else 1 - prob_jaundice
        
        return {
            'result': result,
            'confidence': float(confidence),
            'risk_score': float(prob_jaundice),
            'message': 'Jaundice detected - consult doctor immediately.' if prediction == 1 else 'Normal sclera color.'
        }
    finally:
        os.unlink(tmp_path)
