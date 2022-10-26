import numpy as np
import os
import nibabel as nib
from tqdm import tqdm as tq

img_path = '/home/rakshith/Datasets/Task03_Liver/np_array/images_tr/'
label_path = '/home/rakshith/Datasets/Task03_Liver/np_array/labels_tr/'

image_list = os.listdir(img_path)
label_list = os.listdir(label_path)

for i,img_name in enumerate(image_list):
    print(f'Count: {i} Total: {len(image_list)}')
    img = np.load(img_path+img_name)
    label = np.load(label_path+img_name)
    for slice in tq(range(img.shape[-1])):
        if np.any(label[:,:,slice]):
            label[label>0] = 1
            file_name = img_name.split('.nii')[0]+'_'+str(slice)
            np.save("/home/rakshith/Datasets/Task03_Liver/np_array_slices/images_tr/"+file_name, img[:,:,slice])
            np.save("/home/rakshith/Datasets/Task03_Liver/np_array_slices/labels_tr/"+file_name, label[:,:,slice])
