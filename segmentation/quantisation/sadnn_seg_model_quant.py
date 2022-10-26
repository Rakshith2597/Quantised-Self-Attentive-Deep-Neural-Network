import torch
from torch import nn
import torch.nn.functional as F
import math
from   torchvision import models
import torch.nn.init as init
from torchsummary import summary
from torch.quantization import QuantStub, DeQuantStub

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False,
                 R=3, z_init=0.3, image_size=32):

        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        
        self.kernel_size =  kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divisible by groups. (example: out_channels: 40, groups: 4)"

        max_mask_size = image_size / 2 # TODO(Joe): Our images are all even sizes now so this works but we should force this to be an int, i.e. int(image_size / 2) or image_size // 2

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, self.kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, self.kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.reset_parameters()

    def forward(self, x):

        batch, channels, height, width = x.size()
        max_size = None

        kernel_size = self.kernel_size
        x = self.dequant(x)
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        padded_x = self.quant(padded_x)
        x = self.quant(x)
       
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        # this line is dividing k_out into chunks of kernel_size with stride=self.stride
        k_out = k_out.unfold(2, kernel_size, self.stride).unfold(3, kernel_size, self.stride)
        v_out = v_out.unfold(2, kernel_size, self.stride).unfold(3, kernel_size, self.stride)
       
        rel_h = self.rel_h
        rel_w = self.rel_w
    
        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out_h = self.dequant(k_out_h)
        k_out_w = self.dequant(k_out_w)
        # k_out = torch.cat((k_out_h + rel_h, k_out_w + rel_w), dim=1)
        temp_rel_h = rel_h.unsqueeze(0).repeat(1,1,k_out_h.shape[2],k_out_h.shape[3],1,k_out_h.shape[5])
        temp_rel_w = rel_w.unsqueeze(0).repeat(1,1,k_out_w.shape[2],k_out_w.shape[3],k_out_w.shape[4],1)
      
        # k_out_h - self.dequant(k_out_h)
        # k_out_w - self.dequant(k_out_w)
        # temp_rel_h = self.dequant(temp_rel_h)
        # temp_rel_w = self.dequant(temp_rel_w)
        k_out = torch.cat((k_out_h + temp_rel_h, k_out_w + temp_rel_w), dim=1)
        k_out = self.quant(k_out)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)
      
        q_out = self.dequant(q_out)
        k_out = self.dequant(k_out)
        v_out = self.dequant(v_out)
        out = q_out * k_out       
        out = F.softmax(out, dim=-1)        
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)
        out = self.quant(out)        
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight,
                             mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight,
                             mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight,
                             mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)


class FullAttenSUMNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FullAttenSUMNet, self).__init__()

        self.conv1     = AttentionConv(in_ch,64,3,padding=1)
        self.bn1       = nn.BatchNorm2d(64,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        self.conv2     = AttentionConv(64,128,3,padding=1)   
        self.bn2       = nn.BatchNorm2d(128,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.pool     = nn.MaxPool2d(2, 2, return_indices = True)
        # self.pool     = FP32NonTraceable_maxpool()

        self.conv3a    = AttentionConv(128,256,3,padding=1) 
        self.bn3a       = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        self.conv3b    = AttentionConv(256,256,3,padding=1)
        self.bn3b       = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        # self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)

        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv4a    = AttentionConv(256,512,3,padding=1) 
        self.bn4a       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        self.conv4b    = AttentionConv(512,512,3,padding=1)
        self.bn4b       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        # self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)

        self.dropout2 = nn.Dropout2d(p=0.25)

        self.conv5a    = AttentionConv(512,512,3,padding=1) 
        self.bn5a       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        self.conv5b    = AttentionConv(512,512,3,padding=1)
        self.bn5b       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        # self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)     

        self.unpool4   = nn.MaxUnpool2d(2, 2)
        self.donv5b    = AttentionConv(1024, 512, 3, padding = 1)

        self.donv5a    = AttentionConv(512, 512, 3, padding = 1)

        self.unpool3   = nn.MaxUnpool2d(2, 2)
        self.donv4b    = AttentionConv(1024, 512, 3, padding = 1)

        self.donv4a    = AttentionConv(512, 256, 3, padding = 1)

        self.unpool2   = nn.MaxUnpool2d(2, 2)
        self.donv3b    = AttentionConv(512, 256, 3, padding = 1)

        self.donv3a    = AttentionConv(256,128, 3, padding = 1)

        self.unpool1   = nn.MaxUnpool2d(2, 2)
        self.donv2     = AttentionConv(256, 64, 3, padding = 1)

        self.donv1     = AttentionConv(128, 32, 3, padding = 1)

        self.output    = AttentionConv(32, out_ch, 1)
        self.relu = nn.ReLU(inplace = True)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        conv1          = self.relu(self.bn1(self.conv1(x)))
        conv2          = self.relu(self.bn2(self.conv2(conv1)))
        conv2 = self.dequant(conv2)
        pool1, idxs1   = self.pool(conv2)
        pool1 = self.quant(pool1)        
        conv3a         = self.relu(self.bn3a(self.conv3a(pool1)))
        conv3b         = self.relu(self.bn3b(self.conv3b(conv3a)))
        conv3b = self.dequant(conv3b)
        pool2, idxs2   = self.pool(conv3b)
        pool2 = self.quant(pool2) 
        conv4a         = self.relu(self.bn4a(self.conv4a(self.dropout1(pool2))))
        conv4b         = self.relu(self.bn4b(self.conv4b(conv4a)))
        conv4b = self.dequant(conv4b)
        pool3, idxs3   = self.pool(conv4b)
        pool3 = self.quant(pool3)
        conv5a         = self.relu(self.bn5a(self.conv5a(self.dropout2(pool3))))
        conv5b         = self.relu(self.bn5b(self.conv5b(conv5a)))
        conv5b = self.dequant(conv5b)
        pool4, idxs4   = self.pool(conv5b)
        # pool4 = self.quant(pool4)
        
        unpool4        = torch.cat([self.unpool4(pool4, idxs4), conv5b], 1)
        unpool4 = self.quant(unpool4)
        donv5b         = self.relu(self.donv5b(unpool4))
        donv5a         = self.relu(self.donv5a(donv5b))
        donv5a = self.dequant(donv5a)
        unpool3        = torch.cat([self.unpool4(donv5a, idxs3), conv4b], 1)
        unpool3 = self.quant(unpool3)
        donv4b         = self.relu(self.donv4b(unpool3))
        donv4a         = self.relu(self.donv4a(donv4b))
        donv4a = self.dequant(donv4a)
        unpool2        = torch.cat([self.unpool3(donv4a, idxs2), conv3b], 1)
        unpool2 = self.quant(unpool2)
        donv3b         = self.relu(self.donv3b(unpool2))
        donv3a         = self.relu(self.donv3a(donv3b))
        donv3a = self.dequant(donv3a)
        unpool1        = torch.cat([self.unpool2(donv3a, idxs1), conv2], 1)
        unpool1 = self.quant(unpool1)
        donv2          = self.relu(self.donv2(unpool1))
        donv1          = self.relu(self.donv1(torch.cat([donv2,conv1],1)))
        output         = self.output(donv1)
        output = self.dequant(output)
       
        return output

class FP32NonTraceable_maxpool(nn.Module):
    def __init__(self):
        super(FP32NonTraceable_maxpool, self).__init__()
        self.pool =  nn.MaxPool2d(2, 2, return_indices = True)

    def forward(self, x):
        x, idx = self.pool(x)
        return x, idx
