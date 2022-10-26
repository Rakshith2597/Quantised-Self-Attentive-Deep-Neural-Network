import torch
from self_attn_sumnet import FullAttenSUMNet
from train import test
import os

model = FullAttenSUMNet(in_ch=1, out_ch=2)
model_path = '/home/rakshith/miccai_2022/model_weights/segmentation/ssa_seg_adam_1e4_e15_re/'
checkpoint= torch.load(model_path+'model_best_dice.pt')
model.load_state_dict(checkpoint['state_dict'])
backend = "fbgemm"
model.eval()
model.qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = torch.quantization.prepare(model, inplace=False)
model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
state = {'state_dict': model_static_quantized.state_dict() }
save_path = '/home/rakshith/miccai_2022/model_weights/segmentation/ssa_seg_adam_1e4_e15_re_quantised/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
torch.save(state, save_path+'model_best_dice.pt')
model_name = 'ssa_seg_adam_1e4_e15_re_quantised'
config = {
        "savepath":"/home/rakshith/miccai_2022/model_weights/segmentation/"+model_name+"/",
        "datapath": "/home/rakshith/Datasets/Task03_Liver/np_array_slices/",
        "data_split": "/home/rakshith/miccai_2022/segmentation/train_valid_split.json",
        "out_ch" : 2,
        "model_name":model_name,
        "model" : 'ssa_quant',
        "quantised": 1
    }
test(config, model_static_quantized)