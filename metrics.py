import numpy as np
import os
from tqdm import tqdm  # Import tqdm for progress bar

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define paths
validation_data_dir = 'data/test/'

# Prepare validation data generator
validation_datagen = ImageDataGenerator(rescale=1./255)


validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='rgb',  # Change to RGB
    target_size=(224, 224),  # Match model's expected size
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Load the trained model
model = load_model('emotion_detection_model.h5')

# Compile the model again to avoid the "metrics have yet to be built" warning
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Predict all data at once
print("Predicting on test data...")
y_pred_prob = model.predict(validation_generator, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)

# True labels
y_true = validation_generator.classes

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate precision, recall, and f1-score (for multi-class classification)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Print metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
