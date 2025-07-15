# Unusual Activity Detection Using CNN-LSTM

This project is a deep learning-based system for detecting unusual or suspicious human activities (like violence or vandalism) from video streams using a combination of **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** layers. It features real-time inference support and live alerting mechanisms.

---

## 🔄 Overview

* **Goal:** Detect and classify human activity into `LiftVandalism`, `Violence`, and `NonViolence`.
* **Approach:** Frame-wise feature extraction using CNN followed by temporal modeling using LSTM (LRCN architecture).
* **Frameworks:** TensorFlow, Keras, OpenCV
* **Real-Time:** Live camera/video stream support via socket communication
* **Alerts:** Email (with attached frame) and optional SMS alerts on violence detection

---

## 📁 Project Structure

```bash
unusual-activity-detection/
├── dataset/                      # Original video data (not uploaded)
├── augmentation.py              # Video augmentation logic
├── preprocessing.py             # Frame extraction and .npz dataset creation
├── model_training.py            # CNN-LSTM model creation and training
├── model_testing_live.py        # Live stream receiver and alert system
├── requirements.txt
├── README.md
└── saved_models/                # Trained .h5 models (optional to upload)
```

---

## 🔹 Model Pipeline

### 1. **Preprocessing**

* Frames extracted from videos (fixed 20-frame sequences)
* Normalized and resized to 64x64
* Dataset saved in `.npz` format

### 2. **Model Architecture (LRCN)**

* TimeDistributed Conv2D + BatchNorm + Pooling + Dropout
* Bidirectional LSTM layers
* Final softmax layer for 3-class prediction

### 3. **Training**

* Dataset split: 75% train / 25% validation
* Optimizer: Adam
* Loss: Categorical Crossentropy
* Callbacks: EarlyStopping
* Mixed Precision training for GPU performance boost

---

## 💪 Augmentations Used

Custom video augmentations for better generalization:

* Horizontal Flip
* Brightness Up/Down
* Speed Up/Down
* Gaussian Blur
* Rotation
* Shadow Injection

These were applied to increase the size and diversity of the dataset.

---

## 🔹 Inference and Deployment

### ✈️ **Offline Testing**

* Load any test video file
* Extract frames and classify
* Show predictions + confusion matrix

### 🚀 **Real-Time Streaming (Live Socket Inference)**

* `Sender`: Streams live frames to receiver
* `Receiver`: Receives, buffers, classifies using trained model, and shows results

#### Alerts

* Email alerts with attached frame sent on "Violence" or "LiftVandalism"
* Optional SMS alert via Twilio (can be toggled)

---

## 📊 Results

* **Test Accuracy:** \~99.5%
* **Real-time Inference FPS:** \~15-25 (depending on system)
* **Alerts Sent:** Email (with snapshot)

---

## 📅 Dependencies

Install with:

```bash
pip install -r requirements.txt
```

Or manually:

```txt
tensorflow>=2.9.0
keras>=2.9.0
numpy>=1.21.0
pandas>=1.3.0
opencv-python>=4.5.3
matplotlib>=3.5.0
scikit-learn>=1.0.0
seaborn>=0.11.2
moviepy>=1.0.3
numba
smtplib
email
threading
socket
pickle
twilio
```

---

## 🔗 How to Run

### 1. **Train the Model**

```bash
python model_training.py
```

### 2. **Test Locally (Offline)**

```bash
python preprocessing.py        # Creates dataset
python model_testing_offline.py  # (If separated for local video testing)
```

### 3. **Run Live Detection System**

On Receiver PC:

```bash
python model_testing_live.py
```

On Sender PC (or sender script):

```bash
# Stream webcam or video file using OpenCV + socket
```

---

## 🧰 Future Improvements

* Integrate with CCTV/RTSP streams
* Host as Flask web app with real-time dashboard
* Export alerts to cloud logging / monitoring

---

## 🙏 Credits

* Developed by: Joe Fernandes|Reezann Pereira|Pranali Palav|Shayne Cradozo
* Contact for queries: joecansiofernandes@gmail.com

---

## ⚠️ Notes

* Dataset and trained models are large and excluded from the repo.
* For testing, use your own sample videos or contact the author.
* Keep email/Twilio credentials secure (use environment variables in production).


