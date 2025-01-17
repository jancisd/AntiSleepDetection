import cv2
import os
import tensorflow as tf
from keras.layers import TFSMLayer
import numpy as np
from pygame import mixer
import time

# Initialize sound mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

# Paths to Haar Cascade files
face_cascade_path = '/Users/jancisdelfin/Desktop/project1/Drowsiness_detection/haar-cascade-files/haarcascade_frontalface_alt.xml'
leye_cascade_path = '/Users/jancisdelfin/Desktop/project1/Drowsiness_detection/haar-cascade-files/haarcascade_lefteye_2splits.xml'
reye_cascade_path = '/Users/jancisdelfin/Desktop/project1/Drowsiness_detection/haar-cascade-files/haarcascade_righteye_2splits.xml'

# Validate file paths
for path in [face_cascade_path, leye_cascade_path, reye_cascade_path, 'alarm.wav']:
    if not os.path.exists(path):
        print(f"File not found: {path}")
        exit()

# Load Haar Cascades
face = cv2.CascadeClassifier(face_cascade_path)
leye = cv2.CascadeClassifier(leye_cascade_path)
reye = cv2.CascadeClassifier(reye_cascade_path)

# Load the TensorFlow SavedModel using TFSMLayer
model = TFSMLayer('models/cnnCat2_saved_model', call_endpoint='serving_default')

# Webcam setup
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count, score, thicc = 0, 0, 2
rpred, lpred = [99], [99]
lbl = ['Close', 'Open']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces and eyes
    faces = face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in right_eye:
        r_eye = gray[y:y + h, x:x + w]
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255.0
        r_eye = r_eye.reshape(1, 24, 24, 1)
        
        # Debug output
        model_output = model(r_eye)
        print("Model output (right eye):", model_output)
        
        rpred = np.argmax(model_output['output_0'].numpy(), axis=-1)  # Update key if needed
        print("Right eye prediction:", rpred)
        break

    # Detect left eye
    for (x, y, w, h) in left_eye:
        l_eye = gray[y:y + h, x:x + w]
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255.0
        l_eye = l_eye.reshape(1, 24, 24, 1)
        
        # Debug output
        model_output = model(l_eye)
        print("Model output (left eye):", model_output)
        
        lpred = np.argmax(model_output['output_0'].numpy(), axis=-1)  # Update key if needed
        print("Left eye prediction:", lpred)
        break

    # Drowsiness detection logic
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = max(0, score - 1)
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f'Score: {score}', (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Trigger alarm if score exceeds threshold
    if score > 15:
        cv2.imwrite(os.path.join(os.getcwd(), 'image.jpg'), frame)
        try:
            if not mixer.get_busy():
                sound.play()
        except Exception as e:
            print("Error playing sound:", e)
        thicc = min(16, thicc + 2)
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
    else:
        thicc = max(2, thicc - 2)

    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
