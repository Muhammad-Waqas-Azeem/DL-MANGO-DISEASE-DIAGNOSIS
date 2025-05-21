import tensorflow as tf
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2

# Load trained model
model_path = "D:\\mango\\best_model.keras"
model = tf.keras.models.load_model(model_path)

# Validation dataset path
valid_dir = "C:\\Users\\waqas\\Downloads\\Compressed\\MangoLeafBD Dataset\\MangoLeafBD Dataset"

# Get class names from directory names
class_names = sorted(os.listdir(valid_dir))

# Select 100 random images from validation dataset
random_images = []
random_labels = []

for _ in range(100):
    class_name = random.choice(class_names)  # Pick a random class
    class_path = os.path.join(valid_dir, class_name)
    image_name = random.choice(os.listdir(class_path))  # Pick a random image
    image_path = os.path.join(class_path, image_name)

    # Load image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img / 255.0
    random_images.append(img_array)
    random_labels.append(class_name)

# Convert to numpy array
random_images = np.array(random_images)

# Get predictions
predictions = model.predict(random_images)
predicted_labels = [class_names[np.argmax(pred)] for pred in predictions]

# Plot results in a 10x10 grid
plt.figure(figsize=(20, 20))

for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.imshow(random_images[i])
    actual = random_labels[i]
    predicted = predicted_labels[i]

    # Color coding
    color = "green" if actual == predicted else "red"
    plt.title(f"A: {actual}\nP: {predicted}", color=color, fontsize=6)
    plt.axis("off")

plt.tight_layout()
plt.show()
