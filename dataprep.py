import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomContrast, Rescaling

# Define dataset path
dataset_path = "C:\\Users\\waqas\\OneDrive\\Desktop\\bigmangods_output"  # Change this to your dataset location

# Load dataset
raw_train_dataset = image_dataset_from_directory(
    dataset_path,
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

raw_val_dataset = image_dataset_from_directory(
    dataset_path,
    image_size=(256, 256),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Get class names before transformations
class_names = raw_train_dataset.class_names
print("Classes:", class_names)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomContrast(0.2)
])

# Normalize images
normalization_layer = Rescaling(1./255)

# Apply augmentation only to training dataset
train_dataset = raw_train_dataset.map(lambda x, y: (data_augmentation(normalization_layer(x)), y))
val_dataset = raw_val_dataset.map(lambda x, y: (normalization_layer(x), y))

# Save datasets (New Method)
train_dataset.save("train_dataset")
val_dataset.save("val_dataset")
