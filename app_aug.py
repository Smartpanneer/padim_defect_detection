import os
import cv2
import random
import numpy as np
from torchvision import transforms
from PIL import Image

# ==========================================
# CONFIG
# ==========================================
INPUT_FOLDER = "new_data"
OUTPUT_FOLDER = "train_model_augmented"
AUGMENT_TIMES = 8   # 15 images x 8 = 120 new images

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==========================================
# SAFE AUGMENTATIONS
# ==========================================
augment_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
    transforms.GaussianBlur(kernel_size=3)
])

print("Starting augmentation...")

for file in os.listdir(INPUT_FOLDER):

    input_path = os.path.join(INPUT_FOLDER, file)
    img = cv2.imread(input_path)

    if img is None:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Save original image also
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"orig_{file}"), img)

    # Generate augmentations
    for i in range(AUGMENT_TIMES):
        aug_img = augment_transform(img_pil)

        aug_np = np.array(aug_img)
        aug_bgr = cv2.cvtColor(aug_np, cv2.COLOR_RGB2BGR)

        save_name = f"{os.path.splitext(file)[0]}_aug_{i}.jpg"
        save_path = os.path.join(OUTPUT_FOLDER, save_name)

        cv2.imwrite(save_path, aug_bgr)

print("Augmentation completed.")
print("Augmented images saved in:", OUTPUT_FOLDER)
