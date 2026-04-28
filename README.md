# 👁️ Real-Time Jaundice Detection System

A non-invasive medical diagnostic tool that utilizes **Digital Image Processing** and **Machine Learning (SVM)** to detect jaundice by analyzing the yellowing of the human sclera (the white part of the eye).

## 📌 Project Overview
Jaundice is characterized by elevated bilirubin levels, often manifesting as yellowing of the eyes. This project provides a safer, faster alternative to invasive blood tests by using a camera-based system to classify images into **Normal** or **Jaundice** categories.


---

## 🚀 Key Features
* **Non-Invasive Analysis:** No needles or blood samples required.
* **Multi-Color Space Extraction:** Analyzes images using RGB, HSV, and LAB color models.
* **Specific Channel Targeting:** Utilizes the **LAB b* channel** (Blue-Yellow axis) for high-accuracy yellow detection.
* **SVM Classification:** Powered by a Support Vector Machine for robust binary classification.
* **Real-Time Detection:** Supports live webcam feed for instant feedback.

---

## 🛠️ Technical Workflow
The system follows a rigorous pipeline to ensure clinical relevance:

1.  **Image Acquisition:** Capturing high-resolution images of the eye.
2.  **Preprocessing:** * Resizing to a uniform $224 \times 224$ resolution.
    * White balance correction to eliminate environmental lighting bias.
3.  **Sclera Segmentation:** Isolating the white portion of the eye to remove noise from the iris or eyelids.
4.  **Feature Extraction:** Extracting mean color values from:
    * **RGB:** Basic intensity.
    * **HSV:** To separate color from brightness.
    * **LAB:** Specifically targeting the **b*** channel to measure "yellowness."
5.  **Classification:** An SVM model determines the presence of jaundice based on the extracted feature vector.

---

## 📂 Project Structure
```text
.
├── dataset/             # Organized into /normal and /jaundice
├── models/              # Trained svm_model.pkl and scaler.pkl
├── static/              # CSS and JavaScript for the Web UI
├── templates/           # HTML frontend (index.html)
├── main.py              # Flask/FastAPI application entry point
├── train.py             # Script for model training & evaluation
├── predictor.py         # Logic for processing images and prediction
├── utils.py             # Image processing & segmentation helpers
└── requirements.txt     # List of dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/papxoxo/jaundice-detector.git
   cd jaundice-detector
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   ```bash
   python main.py
   ```

---

## 📊 Model Evaluation
The performance of the SVM classifier is measured using:
* **Accuracy:** Overall correctness.
* **Precision & Recall:** Critical for medical safety (minimizing False Negatives).
* **F1-Score:** The harmonic mean of precision and recall.
* **ROC-AUC:** Evaluating the classifier’s ability to distinguish between classes.




<img width="1408" height="768" alt="Gemini_Generated_Image_pnybi8pnybi8pnyb" src="https://github.com/user-attachments/assets/9b7fb29d-0d89-45c3-a4c5-a5c6cd5a6af0" />

## OUTPUT

<img width="1600" height="1323" alt="WhatsApp Image 2026-04-23 at 12 23 15" src="https://github.com/user-attachments/assets/ea5a99e1-7d88-49e7-9e04-0f76ab5e91a2" />


<img width="1326" height="912" alt="Screenshot 2026-04-28 at 11 28 37 PM" src="https://github.com/user-attachments/assets/8c38bb81-0531-41a8-86d0-68d006afe8b5" />


<img width="1423" height="907" alt="Screenshot 2026-04-28 at 11 28 14 PM" src="https://github.com/user-attachments/assets/b3935467-8756-4022-bd1e-fb9ec610752e" />






---

## ⚠️ Disclaimer
*This project is intended for educational and research purposes only. It is not a substitute for professional medical diagnosis, treatment, or advice. Always consult a healthcare professional for medical concerns.*

---

**Author:** Sudheeksha B H
**License:** MIT
