import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image

class LiverDataLoader(data.Dataset):

    def __init__(self, datapath, file_list, is_transform=True, augmentation=None):

        self.path = datapath
        self.files = file_list
        self.augmentation = augmentation
        self.is_transform = is_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]
        img = np.load(self.path+'images_tr/'+filename)
        img = Image.fromarray(img)
        liver_mask = np.load(self.path+'labels_tr/'+filename)
        liver_mask = Image.fromarray(liver_mask)

        if self.is_transform:
            img, liver_mask = self.transform(img, liver_mask)
            if self.augmentation is not None:
                img, liver_mask = self.augmentation(img, liver_mask)
            labels = torch.cat((1.-liver_mask, liver_mask))

        return img, labels, filename

    def transform(self, img, liver_mask):
        img = transforms.Resize([256,256])(img)
        img = transforms.ToTensor()(img)
        img = img.type(torch.FloatTensor)
        img = transforms.Normalize(-124.8561,312.0627)(img)
        liver_mask = transforms.Resize([256,256])(liver_mask)
        liver_mask = transforms.ToTensor()(liver_mask)
        liver_mask = liver_mask.type(torch.FloatTensor) 

        return img, liver_mask
