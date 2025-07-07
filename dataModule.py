#import albumentations as A
#from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import SkinLesionDataset
from utils import split_dataset

class SkinLesionDataModule(pl.LightningDataModule):
    def __init__(self, image_paths, labels, batch_size=32, num_workers=1, augm_type: str="all"):
        super().__init__()
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augm = augm_type.lower()

        self.train_transform = T.Compose([
            T.RandomHorizontalFlip(),  
            T.RandomVerticalFlip(),  
            T.RandomRotation(degrees=90),
            T.CenterCrop(224),
            T.ToTensor(),  
        ])
        self.test_transform = T.Compose([
            T.CenterCrop(224),  
            T.ToTensor(),  
        ])

    def setup(self, stage=None):
        train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = split_dataset(
            self.image_paths, self.labels
        )
        self.train_dataset = SkinLesionDataset(train_paths, train_labels, transform=self.train_transform, augm_type=self.augm)
        self.val_dataset = SkinLesionDataset(val_paths, val_labels, transform=self.test_transform, augm_type=self.augm)
        self.test_dataset = SkinLesionDataset(test_paths, test_labels, transform=self.test_transform, augm_type=self.augm)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
