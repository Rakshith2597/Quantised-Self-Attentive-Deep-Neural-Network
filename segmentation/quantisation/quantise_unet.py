import torch
import json
from unet import UNet
from train import test
import torchmetrics.functional as TF
import torch.nn.functional as F
import os
from dataloader import LiverDataLoader
from torch.utils import data
from tqdm import tqdm
import torch
from torch.quantization import get_default_qconfig
# Note that this is temporary, we'll expose these functions to torch.quantization after official releasee
from torch.quantization.quantize_fx import prepare_fx, convert_fx

model = UNet(n_channels=1,n_classes=2)
model_path = '/home/rakshith/miccai_2022/model_weights/segmentation/unet_seg_adam_1e4_e15/'
model.load_state_dict(torch.load(model_path+'model_best_seg.pt'))

model_name = 'unet_seg_adam_1e4_e15_re_quantised'
config = {
        "savepath":"/home/rakshith/miccai_2022/model_weights/segmentation/"+model_name+"/",
        "datapath": "/home/rakshith/Datasets/Task03_Liver/np_array_slices/",
        "data_split": "/home/rakshith/miccai_2022/segmentation/train_valid_split.json",
        "out_ch" : 2,
        "model_name":model_name,
        "model" : 'unet_quant',
        "quantised": 1
    }

data_split = config['data_split']

with open('/home/rakshith/miccai_2022/segmentation/train_valid_split.json') as f3:
    data_split = json.load(f3)
file_list = data_split['valid'][:10]
data_path = '/home/rakshith/Datasets/Task03_Liver/np_array_slices/'

testDset = LiverDataLoader(datapath=data_path, file_list=file_list,is_transform=True)

testDataLoader = data.DataLoader(
                            testDset, batch_size=1, drop_last=True,
                            shuffle=False, num_workers=2, pin_memory=True)
model = model.cpu()
model.eval()
qconfig = get_default_qconfig("qnnpack")
qconfig_dict = {"": qconfig}
prepare_custom_config_dict = {
    # option 1
    "non_traceable_module_name": ["up1","up2","up3","up4","ConvTranspose2d"],
}
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target,_ in tqdm(data_loader):
            model(image)

prepared_model = prepare_fx(model, qconfig_dict,prepare_custom_config_dict=prepare_custom_config_dict,)  # fuse modules and insert observers
print('fx model prepared!')
print('Calibration in progress..')
calibrate(prepared_model, testDataLoader)  # run calibration on sample data
print('Calibration completed!')
print('Quantization stated..')
quantized_model_unet = convert_fx(prepared_model)  # convert the calibrated model to a quantized model
print('Quantization completed!')

file_list = data_split['valid']
data_path = '/home/rakshith/Datasets/Task03_Liver/np_array_slices/'

testDset = LiverDataLoader(datapath=data_path, file_list=file_list,is_transform=True)

testDataLoader = data.DataLoader(
                            testDset, batch_size=1, drop_last=True,
                            shuffle=False, num_workers=2, pin_memory=True)

total_dice = 0
dice_cp = []
for i,datasample in tqdm(enumerate(testDataLoader)):
    inputs, mask, fname = datasample
    print(fname)
    
    # inputs = inputs
    # mask = mask
    
    with torch.no_grad():
        seg_out = quantized_model_unet(inputs)
        net_out_sf = F.softmax(seg_out.data, dim=1)

        preds = torch.argmax(net_out_sf, dim=1)
        test_dice = TF.f1_score(preds.flatten().long(), mask[:,1].flatten().long(), num_classes=2, ignore_index=0)
        total_dice += test_dice
        if (i>0) & (i%100 == 0):
            dice_cp.append((i+1,total_dice/(i+1)))
            torch.save(dice_cp,'unet_dice_cp.pt')

print(total_dice/len(testDset))