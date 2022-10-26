import torch
import json
from self_attn_sumnet import SUMNet
from train import test
import torchmetrics.functional as TF
import torch.nn.functional as F
import os
from dataloader import LiverDataLoader
from torch.utils import data
from tqdm import tqdm
import torch

model = SUMNet(in_ch=1,out_ch=2).cuda()
model_path = '/home/rakshith/miccai_2022/model_weights/segmentation/sumnet_seg_adam_1e4_e15/'
model.load_state_dict(torch.load(model_path+'model_best_seg.pt'))

with open('train_valid_split.json') as f3:
    data_split = json.load(f3)
file_list = data_split['valid']
data_path = '/home/rakshith/Datasets/Task03_Liver/np_array_slices/'

testDset = LiverDataLoader(datapath=data_path, file_list=file_list,is_transform=True)

testDataLoader = data.DataLoader(
                            testDset, batch_size=1, drop_last=True,
                            shuffle=False, num_workers=2, pin_memory=True)

total_dice = 0.0
for i,datasample in enumerate(tqdm(testDataLoader)):
    inputs, mask, fname = datasample
    # print(fname)
    
    inputs = inputs.cuda()
    mask = mask.cuda()
    
    with torch.no_grad():
        seg_out = model(inputs)
        net_out_sf = F.softmax(seg_out.data, dim=1)

        preds = torch.argmax(net_out_sf, dim=1)
        test_dice = TF.f1_score(preds.flatten().long(), mask[:,1].flatten().long(), num_classes=2, ignore_index=0)
        total_dice += test_dice
        # if (i>0) & (i%100 == 0):
        #     dice_cp.append((i+1,total_dice/(i+1)))
        #     torch.save(dice_cp,'unet_dice_cp.pt')

print(total_dice/len(testDset))