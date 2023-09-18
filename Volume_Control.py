# fist detection

import cv2
import numpy as np
from playsound import playsound

# Initialize the camera (0 for default camera)
cap = cv2.VideoCapture(0)

# Load the pre-trained hand detection model
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')

# Threshold for detecting fist closure (adjust as needed)
fist_threshold = 10000

fist_closed = False

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to grayscale for hand detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate the area of the detected hand
        hand_area = w * h
        
        # Check if the hand is closed (fist)
        if hand_area < fist_threshold:
            if not fist_closed:
                playsound("alert_sound.wav")  # Replace with your alert sound file
                fist_closed = True
        else:
            fist_closed = False
    
    cv2.imshow("Hand Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
