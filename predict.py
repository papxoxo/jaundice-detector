import cv2
import joblib
from utils import *

model = joblib.load("models/svm_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_image(image_path):
    img = cv2.imread(image_path)
    
    img = resize_image(img)
    img = gray_world_white_balance(img)
    
    sclera, mask = segment_sclera(img)
    features = extract_features(img, mask)
    
    if features is None:
        print("Sclera not detected properly.")
        return
    
    features = scaler.transform([features])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    if prediction == 1:
        print(f"Jaundice Detected (Risk Score: {probability:.2f})")
    else:
        print(f"Normal (Risk Score: {probability:.2f})")

# Example usage
predict_image("test_eye.jpg")