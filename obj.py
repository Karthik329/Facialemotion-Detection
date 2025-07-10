import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(r'C:\Users\karth\Downloads\haarcascade_frontalface_default.xml')

# Load the pre-trained emotion detection model
emotion_model = load_model(r'C:\Users\karth\Downloads\facialemotionmodel.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the live camera
cam = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cam.isOpened():
    print("Camera not working")
    exit()

# Start the live feed
while True:
    # Capture each frame
    ret, frame = cam.read()
    
    # Check if frame is captured successfully
    if not ret:
        print("Can't receive the frame")
        break
    
    # Convert the captured frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the face
        face_roi = gray_frame[y:y + h, x:x + w]
        
        # Preprocess the face ROI for emotion prediction
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Predict the emotion of the face ROI
        emotion_prediction = emotion_model.predict(face_roi)
        max_index = np.argmax(emotion_prediction)
        emotion_label = emotion_labels[max_index]

        # Draw rectangle around face and put text of emotion label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the resulting frame with rectangles and emotion labels
    cv2.imshow('Live Emotion Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
