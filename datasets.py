import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


class DentexDataset(Dataset):
    def __init__(self, image_dir, label_file, augmentations=None):
        self.image_dir = image_dir
        self.label_file = label_file  # Path to the JSON file containing all annotations
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.image_names = [os.path.basename(f) for f in self.image_paths]  # Extract image names
        self.augmentations = augmentations

        # Load the annotations from the single JSON file
        with open(label_file, 'r') as f:
            self.data = json.load(f)

        # Category mapping (category_id_3 -> labels like "Impacted", "Caries", etc.)
        self.category_mapping = {
            0: "Impacted",
            1: "Caries",
            2: "Periapical Lesion",
            3: "Deep Caries"
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = self.image_names[idx]

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = cv2.resize(image, (256, 256))  # Resize for consistency
        image = image.astype(np.float32) / 255.0  # Normalize image to [0, 1]

        # Initialize the mask as all zeros (for 256x256 image)
        mask = np.zeros((256, 256), dtype=np.uint8)  # Mask should be 2D for multiclass segmentation

        # Find the annotation for this image by matching the image name
        image_annotation = next(item for item in self.data['images'] if item['file_name'] == image_name)

        # Process the annotations for this image
        annotations = [anno for anno in self.data['annotations'] if anno['image_id'] == image_annotation['id']]

        for annotation in annotations:
            class_id = annotation['category_id_3']  # Use category_id_3 for the final label (Impacted, Caries, etc.)
            if class_id not in self.category_mapping:
                continue

            # Draw the segmentation polygon on the mask
            segmentation = annotation.get('segmentation', [])
            for polygon in segmentation:
                if isinstance(polygon, list) and len(polygon) > 2:
                    # Convert to numpy array and reshape for fillPoly
                    polygon = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [polygon], class_id)
                else:
                    print("Invalid polygon format:", polygon)

        # Apply augmentations
        if self.augmentations:
            augmented = self.augmentations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).long()

        # Convert mask to one-hot encoding (4 classes)
        mask_one_hot = torch.zeros((4, 256, 256), dtype=torch.float32)
        for i in range(4):
            mask_one_hot[i] = (mask == i).float()

        return image, mask_one_hot



train_dataset = DentexDataset(
    image_dir="DENTEX/training_data/quadrant-enumeration-disease/xrays",
    label_file="DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json",
    augmentations=A.Compose([
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(),
        A.Rotate(limit=30),
        A.Resize(height=256, width=256),  # Direct resize
        ToTensorV2()  # Convert to tensor for PyTorch
    ])
)

# Example: Display a sample from the dataset
sample_image, sample_mask = train_dataset[0]

# Proper conversion: CHW -> HWC
image_np = sample_image.permute(1, 2, 0).cpu().numpy()
image_np = np.clip(image_np, 0, 1)  # ensure values are in [0, 1]

mask_np = sample_mask[0].cpu().numpy()  # Show first class

plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title('Image')

plt.subplot(1, 2, 2)
plt.imshow(mask_np, cmap='gray')
plt.title('Mask (Class 0)')

plt.show()
