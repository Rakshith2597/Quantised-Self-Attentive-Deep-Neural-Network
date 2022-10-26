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

from torch.quantization import get_default_qconfig
# Note that this is temporary, we'll expose these functions to torch.quantization after official releasee
from torch.quantization.quantize_fx import prepare_fx, convert_fx
import json
from torch.utils import data
from dataloader import LiverDataLoader
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

model = FullAttenSUMNet(in_ch=1, out_ch=2)
model_path = '/home/rakshith/miccai_2022/model_weights/segmentation/ssa_seg_adam_1e4_e15_re/'
checkpoint= torch.load(model_path+'model_best_dice.pt')
# checkpoint= torch.load(model_path+'model_last.pt')
model.load_state_dict(checkpoint['state_dict'])

model_name = 'ssa_seg_adam_1e4_e15_re'
config = {
        "savepath":"/home/rakshith/miccai_2022/model_weights/segmentation/"+model_name+"/",
        "datapath": "/home/rakshith/Datasets/Task03_Liver/np_array_slices/",
        "data_split": "/home/rakshith/miccai_2022/segmentation/train_valid_split.json",
        "out_ch" : 2,
        "model_name":model_name,
        "model" : 'ssa_quant',
        "quantised": 1
    }

# data_split = config['data_split']

with open('/home/rakshith/miccai_2022/segmentation/train_valid_split.json') as f3:
    data_split = json.load(f3)
cal_file_list = data_split['valid'][:10]
data_path = '/home/rakshith/Datasets/Task03_Liver/np_array_slices/'

cal_testDset = LiverDataLoader(datapath=data_path, file_list=cal_file_list,is_transform=True)

calDataLoader = data.DataLoader(
                            cal_testDset, batch_size=1, drop_last=True,
                            shuffle=False, num_workers=2, pin_memory=True)
model.eval()
qconfig = get_default_qconfig("qnnpack")
torch.backends.quantized.engine = 'qnnpack'
qconfig_dict = {"": qconfig}
prepare_custom_config_dict = {
    # option 1
    "non_traceable_module_name": ["pool"],
}
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, _,_ in tqdm(data_loader):
            model(image)

prepared_model = prepare_fx(model, qconfig_dict,prepare_custom_config_dict=prepare_custom_config_dict,)  # fuse modules and insert observers
print('fx model prepared!')
print('Calibration in progress..')
calibrate(prepared_model, calDataLoader)  # run calibration on sample data
print('Calibration completed!')
print('Quantization stated..')
quantized_model = convert_fx(prepared_model)  # convert the calibrated model to a quantized model
print('Quantization completed!')

file_list = data_split['valid']
data_path = '/home/rakshith/Datasets/Task03_Liver/np_array_slices/'

testDset = LiverDataLoader(datapath=data_path, file_list=file_list,is_transform=True)

testDataLoader = data.DataLoader(
                            testDset, batch_size=1, drop_last=True,
                            shuffle=False, num_workers=2, pin_memory=True)

total_dice = 0
dice_cp = []
for i,datasample in enumerate(tqdm(testDataLoader)):
    inputs, mask, fname = datasample
    # print(fname)
    
    # inputs = inputs
    # mask = mask
    
    with torch.no_grad():
        seg_out = quantized_model(inputs)
        net_out_sf = F.softmax(seg_out.data, dim=1)

        preds = torch.argmax(net_out_sf, dim=1)
        test_dice = TF.f1_score(preds.flatten().long(), mask[:,1].flatten().long(), num_classes=2, ignore_index=0)
        print(test_dice)
        total_dice += test_dice
#         if (i>0) & (i%100 == 0):
#             dice_cp.append((i+1,total_dice/(i+1)))
#             torch.save(dice_cp,'sadnn_dice_cp.pt')

# print(total_dice/len(testDset))
