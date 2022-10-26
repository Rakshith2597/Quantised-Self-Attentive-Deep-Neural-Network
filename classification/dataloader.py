import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image


class ChestDataLoader(data.Dataset):
    def __init__(self, img_path, label_json, name_list, is_transform=True):

        self.img_path = img_path
        self.label_json = label_json
        self.files = name_list
        self.is_transform = is_transform
        self.transform = transforms.Compose([
            transforms.Resize([512,512]),
            transforms.ToTensor(),
            transforms.Normalize(0.5149,0.2530)]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]
        img = Image.open(self.img_path + filename)
        label_list = self.label_json[filename]
        Atelectasis, Cardiomegaly, Effusion, Infiltration = 0, 0, 0, 0
        Mass, Nodule, Pneumonia, Pneumothorax = 0, 0, 0, 0
        Consolidation, Edema, Emphysema, Fibrosis = 0, 0, 0, 0
        Pleural_Thickening, Hernia= 0, 0

        for label in label_list:
            if label == 'Atelectasis':
                Atelectasis = 1
            elif label == 'Cardiomegaly':
                Cardiomegaly = 1
            elif label == 'Effusion':
                Effusion = 1
            elif label == 'Infiltration':
                Infiltration = 1
            elif label == 'Mass':
                Mass = 1
            elif label == 'Nodule':
                Nodule = 1
            elif label == 'Pneumonia':
                Pneumonia = 1
            elif label == 'Pneumothorax':
                Pneumothorax = 1
            elif label == 'Consolidation':
                Consolidation = 1
            elif label == 'Edema':
                Edema = 1
            elif label == 'Emphysema':
                Emphysema = 1
            elif label == 'Fibrosis':
                Fibrosis = 1
            elif label == 'Pleural_Thickening':
                Pleural_Thickening = 1
            elif label == 'Hernia':
                Hernia = 1
            else:
                pass

        label_final = [Atelectasis, Cardiomegaly, Effusion, Infiltration,
                        Mass, Nodule, Pneumonia, Pneumothorax,
                        Consolidation, Edema, Emphysema, Fibrosis,
                        Pleural_Thickening, Hernia]
        label_array = np.array(label_final)
        label_tensor = torch.FloatTensor(label_array)

        if self.is_transform:
            img = self.transform(img)
            img = img.type(torch.FloatTensor)

        return img, label_tensor
