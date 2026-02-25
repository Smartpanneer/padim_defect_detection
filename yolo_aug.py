import os
import cv2
import random
import numpy as np
from torchvision import transforms
from PIL import Image
import torch

# ==========================================
# CONFIG
# ==========================================
IMAGE_FOLDER = "/home/aioty-gpu-server-1/padim/dataset/images"
LABEL_FOLDER = "/home/aioty-gpu-server-1/padim/dataset/labels"

OUTPUT_IMAGE_FOLDER = IMAGE_FOLDER
OUTPUT_LABEL_FOLDER = LABEL_FOLDER

AUGMENT_IMAGES = 20
AUGMENT_TIMES = 5   # per image

os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)

# ==========================================
# SAFE AUGMENTATIONS (no rotation, small translate only)
# ==========================================
augment_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3)
])

# ==========================================
# Load 20 random images
# ==========================================
all_images = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith((".jpg", ".png"))]
selected_images = random.sample(all_images, min(AUGMENT_IMAGES, len(all_images)))

print(f"Augmenting {len(selected_images)} images...")

for file in selected_images:

    image_path = os.path.join(IMAGE_FOLDER, file)
    label_path = os.path.join(LABEL_FOLDER, os.path.splitext(file)[0] + ".txt")

    img = cv2.imread(image_path)
    if img is None:
        continue

    h, w = img.shape[:2]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Read YOLO labels
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels = f.readlines()
    else:
        labels = []

    for i in range(AUGMENT_TIMES):

        aug_img = augment_transform(img_pil)

        aug_np = np.array(aug_img)
        aug_bgr = cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR)

        new_image_name = f"{os.path.splitext(file)[0]}_aug_{i}.jpg"
        new_label_name = f"{os.path.splitext(file)[0]}_aug_{i}.txt"

        cv2.imwrite(os.path.join(OUTPUT_IMAGE_FOLDER, new_image_name), aug_bgr)

        # Save same labels (safe because we didn't change geometry)
        with open(os.path.join(OUTPUT_LABEL_FOLDER, new_label_name), "w") as f:
            for line in labels:
                f.write(line)

print("✅ Safe augmentation completed.")
