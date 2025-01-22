
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Disable TensorFlow oneDNN optimizations


# Load the pre-trained model
model = load_model('model_fil_100.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Labels dictionary
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the image
frame = cv2.imread("1.jpeg")
if frame is None:
    print("Error: Unable to load image. Check the file path or format.")
    exit()

# Maintain aspect ratio while resizing
height, width = frame.shape[:2]
max_dimension = max(height, width)
scale_factor = 600 / max_dimension
new_size = (int(width * scale_factor), int(height * scale_factor))
frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))

if len(faces) == 0:
    print("No faces detected.")
else:
    for x, y, w, h in faces:
        # Extract and preprocess the face region
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48), interpolation=cv2.INTER_AREA)
        normalize = resized / 255.0  # Normalization
        reshaped = np.reshape(normalize, (1, 48, 48, 1))

        # Make prediction
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        # Draw bounding box and label
        margin=20
        confidence = np.max(result) * 100
        if confidence > 50:  # Confidence threshold
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box
            cv2.rectangle(frame, (x, y - 40), (x + w+margin, y), (0, 255, 0), -1)  # Background for label
            cv2.putText(
                frame,
                f"{labels_dict[label]})",  # Label with confidence
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
        else:
            print(f"Low confidence for detection at [{x}, {y}, {w}, {h}].")

# Display the processed frame
cv2.imshow("Emotion Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


            

