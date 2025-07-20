# 😊 Real-Time Emotion Detection using MediaPipe & CNN

A real-time **emotion detection system** that uses **MediaPipe** for facial landmark detection and a **Convolutional Neural Network (CNN)** to classify emotions such as *Happy, Sad, Angry, Surprised, Neutral*, etc.

---

## 🎯 Objective

Detect human emotions from live webcam footage based on facial expressions using:
- **MediaPipe Face Mesh** for feature extraction
- **Custom CNN model** for classification
- **OpenCV** for real-time webcam interface

---

## 🧠 Emotion Categories

The model is trained to classify the following emotions:
- Happy 😊  
- Sad 😢  
- Angry 😠  
- Surprised 😮  
- Neutral 😐  
- Fear 😨  
- Disgust 🤢 *(optional)*

---

## 🧪 Formula (Model Flow)

```text
Face → MediaPipe → 468 Landmark Points → Flatten/Normalize → CNN Model → Emotion Label
