import argparse
import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from utils import resize_image, gray_world_white_balance, segment_sclera, extract_features


MODEL_PATH = "models/svm_model.pkl"
SCALER_PATH = "models/scaler.pkl"
DATASET_PATH = "dataset"


# ==================================================
# TRAINING FUNCTION
# ==================================================
def train_model():
    print("Loading dataset...")

    X = []
    y = []

    for label in ["normal", "jaundice"]:
        class_path = os.path.join(DATASET_PATH, label)
        class_label = 0 if label == "normal" else 1

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = resize_image(img)
            img = gray_world_white_balance(img)

            sclera, mask = segment_sclera(img)
            features = extract_features(img, mask)

            if features is not None:
                X.append(features)
                y.append(class_label)

    X = np.array(X)
    y = np.array(y)

    print(f"Total samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Normal images:", len(os.listdir("dataset/normal")))
    print("Jaundice images:", len(os.listdir("dataset/jaundice")))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel="rbf", probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n=== Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\nModel and scaler saved successfully.")


# ==================================================
# IMAGE PREDICTION
# ==================================================
def predict_image(image_path):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    img = cv2.imread(image_path)

    if img is None:
        print("Invalid image path.")
        return

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

    print("\n=== Prediction Result ===")
    if prediction == 1:
        print(f"⚠ Jaundice Detected | Risk Score: {probability:.2f}")
    else:
        print(f"✔ Normal | Risk Score: {probability:.2f}")


# ==================================================
# REAL-TIME CAMERA MODE
# ==================================================
def camera_mode():
    import cv2
    import joblib
    from utils import resize_image, gray_world_white_balance, segment_sclera, extract_features

    print("Loading trained model...")

    model = joblib.load("models/svm_model.pkl")
    scaler = joblib.load("models/scaler.pkl")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not accessible.")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = resize_image(frame)
        img = gray_world_white_balance(img)

        sclera, mask = segment_sclera(img)
        features = extract_features(img, mask)

        label = "Detecting..."

        if features is not None:
            features = scaler.transform([features])

            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1]

            if prediction == 1:
                label = f"JAUNDICE ({probability:.2f})"
            else:
                label = f"NORMAL ({1-probability:.2f})"

        cv2.putText(frame,
                    label,
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.imshow("Neonatal Jaundice Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==================================================
# MAIN ENTRY POINT
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sclera-Based Neonatal Jaundice Detection"
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "predict", "camera"],
        help="Operation mode"
    )

    parser.add_argument(
        "--image",
        help="Path to input image (required for predict mode)"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model()

    elif args.mode == "predict":
        if not args.image:
            print("Please provide --image path for prediction mode.")
        else:
            predict_image(args.image)

    elif args.mode == "camera":
        camera_mode()