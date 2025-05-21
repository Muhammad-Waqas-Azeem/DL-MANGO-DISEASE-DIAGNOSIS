import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model_path = "D:\\mango\\best_model.keras"
model = tf.keras.models.load_model(model_path)

# Class labels (update with actual class names)
class_names = [
    "Anthracnose", 
    "Bacterial Canker", 
    "Cutting Weevil", 
    "Die Back", 
    "Gall Midge", 
    "Healthy", 
    "Powdery Mildew", 
    "Sooty Mould"
]

# Path to external image (Change this to your actual image path)
image_path = "C:\\Users\\waqas\\OneDrive\\Desktop\\Gall-midge.png"

# Load and preprocess the image
img = cv2.imread(image_path)
img = cv2.resize(img, (256, 256))  # Resize to match model input
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
img = img / 255.0  # Normalize pixel values
img_array = np.expand_dims(img, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

# Output result
print(f"Predicted Class: {predicted_class}")
print(f"🎯 Predicted Class: {predicted_class}")
