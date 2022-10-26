import torch
from self_attn_sumnet import SUMNet
from train import test
import os

model = SUMNet(in_ch=1,out_ch=2)
model_path = '/home/rakshith/miccai_2022/model_weights/segmentation/sumnet_seg_adam_1e4_e15/'
model.load_state_dict(torch.load(model_path+'model_best_seg.pt'))
backend = "fbgemm"
model.eval()
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized_sumnet = torch.quantization.prepare(model, inplace=False)
model_static_quantized_sumnet = torch.quantization.convert(model_static_quantized_sumnet, inplace=False)
model_name = 'sumnet_seg_adam_1e4_e15_re_quantised'
state = {'state_dict': model_static_quantized_sumnet.state_dict()}
save_path = '/home/rakshith/miccai_2022/model_weights/segmentation/unet_seg_adam_1e4_e15_re_quantised/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
torch.save(state, save_path+'model_best_dice.pt')
config = {
        "savepath":"/home/rakshith/miccai_2022/model_weights/segmentation/"+model_name+"/",
        "datapath": "/home/rakshith/Datasets/Task03_Liver/np_array_slices/",
        "data_split": "/home/rakshith/miccai_2022/segmentation/train_valid_split.json",
        "out_ch" : 2,
        "model_name":model_name,
        "model" : 'sumnet_quant',
        "quantised": 1
    }
test(config, model_static_quantized_sumnet)
