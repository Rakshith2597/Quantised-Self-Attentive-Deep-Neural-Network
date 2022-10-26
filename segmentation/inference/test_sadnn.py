import torch
import json
import torch.nn.functional as F
# from self_attn_sumnet import FullAttenSUMNet, SUMNet
# from unet import UNet
from sasa_einsum_sumnet import FullAttenSUMNet
from train import test
from tqdm import tqdm
import torchmetrics.functional as TF
from dataloader import LiverDataLoader
from torch.utils import data

model = FullAttenSUMNet(in_ch=1, out_ch=2).cuda()
model_path = '/home/rakshith/miccai_2022/model_weights/segmentation/ssa_seg_adam_1e4_e15_re/'
checkpoint= torch.load(model_path+'model_best_dice.pt')
# checkpoint= torch.load(model_path+'model_last.pt')
model.load_state_dict(checkpoint['state_dict'])

with open('/home/rakshith/miccai_2022/segmentation/train_valid_split.json') as f3:
    data_split = json.load(f3)
file_list = data_split['valid']
data_path = '/home/rakshith/Datasets/Task03_Liver/np_array_slices/'

testDset = LiverDataLoader(datapath=data_path, file_list=file_list,is_transform=True)

testDataLoader = data.DataLoader(
                            testDset, batch_size=1, drop_last=True,
                            shuffle=False, num_workers=2, pin_memory=True)

model.eval()
total_dice = 0.0
for i,datasample in enumerate(tqdm(testDataLoader)):
    inputs, mask, fname = datasample
    # print(fname)
    
    # inputs = inputs
    # mask = mask
    
    with torch.no_grad():
        seg_out = model(inputs.cuda())
        net_out_sf = F.softmax(seg_out.data, dim=1)

        preds = torch.argmax(net_out_sf, dim=1)
        test_dice = TF.f1_score(preds.flatten().long(), mask[:,1].flatten().long().cuda(), num_classes=2, ignore_index=0)
        print(test_dice)
#         total_dice += test_dice

# print(total_dice/len(testDset))