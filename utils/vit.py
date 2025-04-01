import torch
import timm
import torch.nn as nn

from torch.utils.data import Dataset
from PIL import Image
import os
import glob

from utils.vit_utils import VitUtilities

class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(ViTClassifier, self).__init__()

        # Load Pretrained ViT-B/16 MAE Model
        self.vit = timm.create_model("vit_base_patch16_224.mae", pretrained=True)

        # Get number of features from ViT model
        num_features = self.vit.embed_dim  # Correct way to extract feature size

        # Replace classification head
        self.vit.head = nn.Sequential(
            nn.Dropout(dropout_rate),  
            nn.Linear(num_features, num_classes)  
        )

    def forward(self, x):
        return self.vit(x)

    def extract_features(self, x):
        return self.vit.forward_features(x)

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, CLASS_NAMES):
        self.image_paths = []
        self.labels = []

        for class_name, label in CLASS_NAMES.items():
            class_dir = os.path.join(root_dir, class_name)
            image_files = glob.glob(os.path.join(class_dir, "*.png"))  
            self.image_paths.extend(image_files)
            self.labels.extend([label] * len(image_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")  
        transform = VitUtilities.get_transforms(label)  
        image = transform(image)

        return image, torch.tensor(label, dtype=torch.long)