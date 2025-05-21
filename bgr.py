import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from u2net import U2NET  # Import the model from u2net.py

# -------- Load U2NET model --------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize U2Net model
model = U2NET(3, 1)  # 3 input channels (RGB), 1 output channel (binary mask)
model.load_state_dict(torch.load('u2net.pth', map_location=device))  # Load the model
model.to(device)
model.eval()
print("✅ U-2-Net model loaded successfully!")

# -------- Image Preprocessing --------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 320)),  # Resize image to match input size for U2Net
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Standard normalization
])

# -------- Post-Processing for Transparent Background --------
def post_process(pred, orig_image):
    pred = pred.squeeze().cpu().data.numpy()  # Remove extra dimensions
    pred = (pred - pred.min()) / (pred.max() - pred.min())  # Normalize mask between 0-1
    mask = (pred * 255).astype(np.uint8)  # Convert to 0-255 range
    mask = cv2.resize(mask, (orig_image.width, orig_image.height))  # Resize to original image size

    orig_np = np.array(orig_image)
    if orig_np.shape[2] == 3:
        orig_np = np.dstack((orig_np, np.ones((orig_np.shape[0], orig_np.shape[1]), dtype=np.uint8) * 255))

    orig_np[:, :, 3] = mask  # Set mask as alpha channel
    result = Image.fromarray(orig_np)  # Convert back to PIL image
    return result

# -------- Main Image Processing Loop --------
input_folder = 'C:\\Users\\waqas\\OneDrive\\Desktop\\bigmangods\\Anthracnose'   # Folder with  leaf images
output_folder = 'C:\\Users\\waqas\\OneDrive\\Desktop\\bigmangods\\RbgAnthracnose' # Folder to save output images
os.makedirs(output_folder, exist_ok=True)

print("🚀 Starting background removal...")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
        filepath = os.path.join(input_folder, filename)

        # Load image
        orig_image = Image.open(filepath).convert('RGB')
        input_tensor = transform(orig_image).unsqueeze(0).to(device)

        # Perform prediction
        with torch.no_grad():
            d1, _, _, _, _, _, _ = model(input_tensor)
            pred = d1[:, 0, :, :]  # Only take the first output (the mask)

        # Post-process and save output image with transparent background
        result = post_process(pred, orig_image)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '_transparent.png')
        result.save(output_path)

        print(f"✅ Saved: {output_path}")

print("🎉 All images processed successfully!")
