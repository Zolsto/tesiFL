import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

class SkinLesionDataset(Dataset):
    def __init__(self, image_paths: list, labels: list, transform, augm_type: str="all"):
        self.image_paths = image_paths
        self.labels = labels
        self.augm = augm_type.lower()
        if self.augm=="selective":
            self.low_samples_classes = ["actinic keratosis", "seborrheic keratosis", "squamous cell carcinoma"]
        self.base_tr = T.Compose([T.CenterCrop(224), T.ToTensor()])
        self.transform = transform
        self.targets = self.labels
        self.classes = ["actinic keratosis", "basal cell carcinoma", "melanoma", "nevus", "seborrheic keratosis", "squamous cell carcinoma"]
        self.class_to_idx = {
            0: "actinic keratosis",
            1: "basal cell carcinoma",
            2: "melanoma",
            3: "nevus",
            4: "seborrheic keratosis",
            5: "squamous cell carcinoma"}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        if self.augm!="no":
            if self.augm=="selective":
                if self.class_to_name[label] in self.low_samples_classes:
                    image = self.transform(image)
                else:
                    image = self.base_tr(image)
            else:
                image = self.transform(image)
            
        label = torch.tensor(label, dtype=torch.long)
        return image, label
