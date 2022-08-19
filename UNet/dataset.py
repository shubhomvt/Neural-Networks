import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir) #list all the files in that folder

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img_path = os.path.join(self.image_dir, self.images[index] ) #join image path and folder path
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))# np.array that is numpy here because we
        # will be using alby mutation library which if u are using pills needs to be converted to numpy array    
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #since it will be grayscale and thats how u do it for PIL we pul "L"
        mask[mask==255.0] = 1.0 # we are doing this since we will be using sigmoid as our last activation
        #indicating the prob

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask