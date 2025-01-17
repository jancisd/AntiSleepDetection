# AntiSleepDetection

AntiSleepDetection is a real-time drowsiness detection system that uses computer vision and deep learning to detect if a person's eyes are closed for an extended period. If drowsiness is detected, an alarm is triggered to alert the user, making it ideal for applications like driver safety.

---

## Features
- Real-time face and eye detection using OpenCV.
- Deep learning-based eye state classification (open or closed).
- Alarm system to alert the user in case of drowsiness.


---

## How It Works
1. The system captures live video from a webcam.
2. **Haar Cascades** detect the face and eyes in each frame.
3. A pre-trained TensorFlow model classifies the eye state (open or closed).
4. If both eyes remain closed for a threshold period, an alarm is triggered.

---


