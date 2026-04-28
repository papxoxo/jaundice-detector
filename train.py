import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from utils import *

dataset_path = "dataset"

X = []
y = []

for label in ["normal", "jaundice"]:
    class_path = os.path.join(dataset_path, label)
    class_label = 0 if label == "normal" else 1
    
    for file in os.listdir(class_path):
        img_path = os.path.join(class_path, file)
        img = cv2.imread(img_path)
        
        img = resize_image(img)
        img = gray_world_white_balance(img)
        
        sclera, mask = segment_sclera(img)
        features = extract_features(img, mask)
        
        if features is not None:
            X.append(features)
            y.append(class_label)

X = np.array(X)
y = np.array(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,  random_state=42
)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM Model
model = SVC(kernel="rbf", probability=True, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# Save model
joblib.dump(model, "models/svm_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model saved successfully.")