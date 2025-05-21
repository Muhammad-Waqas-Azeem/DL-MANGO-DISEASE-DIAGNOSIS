import os
import shutil
import random

# Paths
original_data_dir = "C:\\Users\\waqas\\OneDrive\\Desktop\\bigmangods_output"  # Change this to your dataset folder
base_dir = "C:\\Users\\waqas\\OneDrive\\Desktop\\bmdstv"  # Where train/valid will be stored

train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")

# Create directories if not exist
for split in [train_dir, valid_dir]:
    os.makedirs(split, exist_ok=True)

# Get all class folders (diseases + healthy)
classes = os.listdir(original_data_dir)

# Split data (80% train, 20% valid)
for class_name in classes:
    class_path = os.path.join(original_data_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # Skip non-folder files

    images = os.listdir(class_path)
    random.shuffle(images)

    split_idx = int(len(images) * 0.8)  # 80% train, 20% valid
    train_images = images[:split_idx]
    valid_images = images[split_idx:]

    # Create class subdirectories
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(valid_dir, class_name), exist_ok=True)

    # Move files
    for img in train_images:
        shutil.move(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    for img in valid_images:
        shutil.move(os.path.join(class_path, img), os.path.join(valid_dir, class_name, img))

print("✅ Dataset split into train & valid successfully!")
