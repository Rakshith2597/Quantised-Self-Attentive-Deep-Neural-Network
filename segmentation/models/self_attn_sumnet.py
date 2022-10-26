import torch
from torch import nn
import torch.nn.functional as F
import math
from   torchvision import models
import torch.nn.init as init
from torchsummary import summary
from torch.quantization import QuantStub, DeQuantStub


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, height, width, -1, self.out_channels // self.groups)
#         p_out = p_out.contiguous().view(batch, self.groups, height, width, -1, self.out_channels // self.groups)
        v_out = v_out.contiguous().view(batch, self.groups, height, width, -1, self.out_channels // self.groups)

        q_out = q_out.contiguous().view(batch, self.groups, height, width, 1, self.out_channels // self.groups)

        out = torch.matmul(q_out, (k_out).transpose(-1, -2))
        out = F.softmax(out, dim=-1)

        out = torch.matmul(out, v_out).view(batch, -1, height, width)
        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class FullAttenSUMNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FullAttenSUMNet, self).__init__()

        self.conv1     = AttentionConv(in_ch,64,3,padding=1)
        self.bn1       = nn.BatchNorm2d(64,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        self.conv2     = AttentionConv(64,128,3,padding=1)   
        self.bn2       = nn.BatchNorm2d(128,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.pool1     = nn.MaxPool2d(2, 2, return_indices = True)
        # self.pool     = FP32NonTraceable_maxpool()

        self.conv3a    = AttentionConv(128,256,3,padding=1) 
        self.bn3a       = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        self.conv3b    = AttentionConv(256,256,3,padding=1)
        self.bn3b       = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.pool2     = nn.MaxPool2d(2, 2, return_indices = True)

        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv4a    = AttentionConv(256,512,3,padding=1) 
        self.bn4a       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        self.conv4b    = AttentionConv(512,512,3,padding=1)
        self.bn4b       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.pool3     = nn.MaxPool2d(2, 2, return_indices = True)

        self.dropout2 = nn.Dropout2d(p=0.25)

        self.conv5a    = AttentionConv(512,512,3,padding=1) 
        self.bn5a       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)

        self.conv5b    = AttentionConv(512,512,3,padding=1)
        self.bn5b       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.pool4     = nn.MaxPool2d(2, 2, return_indices = True)     

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
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()

    def forward(self, x):
        # x = self.quant(x)
        conv1          = self.relu(self.bn1(self.conv1(x)))
        conv2          = self.relu(self.bn2(self.conv2(conv1)))
        # conv2 = self.dequant(conv2)
        pool1, idxs1   = self.pool1(conv2)
        # pool1 = self.quant(pool1)        
        conv3a         = self.relu(self.bn3a(self.conv3a(pool1)))
        conv3b         = self.relu(self.bn3b(self.conv3b(conv3a)))
        # conv3b = self.dequant(conv3b)
        pool2, idxs2   = self.pool2(conv3b)
        # pool2 = self.quant(pool2) 
        conv4a         = self.relu(self.bn4a(self.conv4a(self.dropout1(pool2))))
        conv4b         = self.relu(self.bn4b(self.conv4b(conv4a)))
        # conv4b = self.dequant(conv4b)
        pool3, idxs3   = self.pool3(conv4b)
        # pool3 = self.quant(pool3)
        conv5a         = self.relu(self.bn5a(self.conv5a(self.dropout2(pool3))))
        conv5b         = self.relu(self.bn5b(self.conv5b(conv5a)))
        # conv5b = self.dequant(conv5b)
        pool4, idxs4   = self.pool4(conv5b)
        # pool4 = self.dequant(pool4)
        
        unpool4        = torch.cat([self.unpool4(pool4, idxs4), conv5b], 1)
        # unpool4 = self.quant(unpool4)
        donv5b         = self.relu(self.donv5b(unpool4))
        donv5a         = self.relu(self.donv5a(donv5b))
        # donv5a = self.dequant(donv5a)
        unpool3        = torch.cat([self.unpool4(donv5a, idxs3), conv4b], 1)
        # unpool3 = self.quant(unpool3)
        donv4b         = self.relu(self.donv4b(unpool3))
        donv4a         = self.relu(self.donv4a(donv4b))
        # donv4a = self.dequant(donv4a)
        unpool2        = torch.cat([self.unpool3(donv4a, idxs2), conv3b], 1)
        # unpool2 = self.quant(unpool2)
        donv3b         = self.relu(self.donv3b(unpool2))
        donv3a         = self.relu(self.donv3a(donv3b))
        # donv3a = self.dequant(donv3a)
        unpool1        = torch.cat([self.unpool2(donv3a, idxs1), conv2], 1)
        # unpool1 = self.quant(unpool1)
        donv2          = self.relu(self.donv2(unpool1))
        donv1          = self.relu(self.donv1(torch.cat([donv2,conv1],1)))
        output         = self.output(donv1)
        # output = self.dequant(output)
       
        return output

class FP32NonTraceable_maxpool(nn.Module):
    def __init__(self):
        super(FP32NonTraceable_maxpool, self).__init__()
        self.pool =  nn.MaxPool2d(2, 2, return_indices = True)

    def forward(self, x):
        x, idx = self.pool(x)
        return x, idx

class SUMNet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(SUMNet, self).__init__()
        
        self.conv1     = nn.Conv2d(in_ch,64,3,padding=1)
        self.bn1       = nn.BatchNorm2d(64,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.conv2     = nn.Conv2d(64,128,3,padding=1)   
        self.bn2       = nn.BatchNorm2d(128,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.pool     = FP32NonTraceable_maxpool()
        self.conv3a    = nn.Conv2d(128,256,3,padding=1) 
        self.bn3a       = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.conv3b    = nn.Conv2d(256,256,3,padding=1)
        self.bn3b       = nn.BatchNorm2d(256,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        # self.pool2     = FP32NonTraceable_maxpool()
        self.conv4a    = nn.Conv2d(256,512,3,padding=1) 
        self.bn4a       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.conv4b    = nn.Conv2d(512,512,3,padding=1)
        self.bn4b       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        # self.pool3     = FP32NonTraceable_maxpool()
        self.conv5a    = nn.Conv2d(512,512,3,padding=1) 
        self.bn5a       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        self.conv5b    = nn.Conv2d(512,512,3,padding=1)
        self.bn5b       = nn.BatchNorm2d(512,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True)
        # self.pool4     = FP32NonTraceable_maxpool()
        
        self.unpool4   = nn.MaxUnpool2d(2, 2)
        self.donv5b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.donv5a    = nn.Conv2d(512, 512, 3, padding = 1)
        self.unpool3   = nn.MaxUnpool2d(2, 2)
        self.donv4b    = nn.Conv2d(1024, 512, 3, padding = 1)
        self.donv4a    = nn.Conv2d(512, 256, 3, padding = 1)
        self.unpool2   = nn.MaxUnpool2d(2, 2)
        self.donv3b    = nn.Conv2d(512, 256, 3, padding = 1)
        self.donv3a    = nn.Conv2d(256,128, 3, padding = 1)
        self.unpool1   = nn.MaxUnpool2d(2, 2)
        self.donv2     = nn.Conv2d(256, 64, 3, padding = 1)
        self.donv1     = nn.Conv2d(128, 32, 3, padding = 1)
        self.output    = nn.Conv2d(32, out_ch, 1)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self, x):
        x = self.quant(x)
        conv1          = self.relu(self.bn1(self.conv1(x)))
        conv2          = self.relu(self.bn2(self.conv2(conv1)))
        # conv2 = self.dequant(conv2)
        pool1, idxs1   = self.pool(conv2)
        # pool1 = self.quant(pool1)        
        conv3a         = self.relu(self.bn3a(self.conv3a(pool1)))
        conv3b         = self.relu(self.bn3b(self.conv3b(conv3a)))
        # conv3b = self.dequant(conv3b)
        pool2, idxs2   = self.pool(conv3b)
        # pool2 = self.quant(pool2) 
        conv4a         = self.relu(self.bn4a(self.conv4a(pool2)))
        conv4b         = self.relu(self.bn4b(self.conv4b(conv4a)))
        # conv4b = self.dequant(conv4b)
        pool3, idxs3   = self.pool(conv4b)
        # pool3 = self.quant(pool3)
        conv5a         = self.relu(self.bn5a(self.conv5a(pool3)))
        conv5b         = self.relu(self.bn5b(self.conv5b(conv5a)))
        # conv5b = self.dequant(conv5b)
        pool4, idxs4   = self.pool(conv5b)
        # pool4 = self.dequant(pool4)
        
        unpool4        = torch.cat([self.unpool4(pool4, idxs4), conv5b], 1)
        # unpool4 = self.quant(unpool4)
        donv5b         = F.relu(self.donv5b(unpool4))
        donv5a         = self.relu(self.donv5a(donv5b))
        # donv5a = self.dequant(donv5a)
        unpool3        = torch.cat([self.unpool4(donv5a, idxs3), conv4b], 1)
        # unpool3 = self.quant(unpool3)
        donv4b         = self.relu(self.donv4b(unpool3))
        donv4a         = self.relu(self.donv4a(donv4b))
        # donv4a = self.dequant(donv4a)
        unpool2        = torch.cat([self.unpool3(donv4a, idxs2), conv3b], 1)
        # unpool2 = self.quant(unpool2)
        donv3b         = self.relu(self.donv3b(unpool2))
        donv3a         = self.relu(self.donv3a(donv3b))
        # donv3a = self.dequant(donv3a)
        unpool1        = torch.cat([self.unpool2(donv3a, idxs1), conv2], 1)
        # unpool1 = self.quant(unpool1)
        donv2          = self.relu(self.donv2(unpool1))
        donv1          = self.relu(self.donv1(torch.cat([donv2,conv1],1)))
        output         = self.output(donv1)
        output = self.dequant(output)     
       
        return output

def output_shape(inputs, kernel_size, padding, stride, dilation):
    return [(size + padding * 2 - dilation * (kernel_size - 1) - 1) // stride + 1 for size in inputs.shape[-2:]]


class SelfAttention(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, bias=False):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=bias)

        self.row_embeddings = nn.Parameter(torch.randn(out_channels // 2, kernel_size))
        self.col_embeddings = nn.Parameter(torch.randn(out_channels // 2, kernel_size))

        self.unfold1 = nn.Unfold(kernel_size=1, stride=stride)
        self.unfold2 = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.unfold3 = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)

    def forward(self, inputs):

        queries = self.conv1(inputs)
        queries = self.unfold1(queries)
        queries = queries.reshape(queries.size(0), self.groups, self.out_channels // self.groups, -1, queries.size(-1))
        queries = queries.permute(0, 4, 1, 2, 3) # Query: [B, N, G, C // G, 1]

        keys = self.conv2(inputs)
        keys = self.unfold2(keys)
        keys = keys.reshape(keys.size(0), self.groups, self.out_channels // self.groups, -1, keys.size(-1))
        keys = keys.permute(0, 4, 1, 2, 3) # Key: [B, N, G, C // G, K^2]

        row_embeddings = self.row_embeddings.unsqueeze(-1).expand(-1, -1, self.kernel_size)
        col_embeddings = self.col_embeddings.unsqueeze(-2).expand(-1, self.kernel_size, -1)
        embeddings = torch.cat((row_embeddings, col_embeddings), dim=0)
        embeddings = embeddings.reshape(self.groups, self.out_channels // self.groups, -1)
        embeddings = embeddings.unsqueeze(0).unsqueeze(1) # Embedding: [1, 1, G, C // G, K^2]

        attentions = torch.matmul(torch.transpose(queries, -2, -1), keys + embeddings)
        attentions = nn.functional.softmax(attentions, dim=-1) # Attention: [B, N, G, 1, K^2]

        values = self.conv3(inputs)
        values = self.unfold3(values)
        values = values.reshape(values.size(0), self.groups, self.out_channels // self.groups, -1, values.size(-1))
        values = values.permute(0, 4, 1, 2, 3) # Value: [B, N, G, C // G, K^2]

        outputs = torch.matmul(values, torch.transpose(attentions, -2, -1)) # Self-Attention: [B, N, G, C // G, 1]
        outputs = outputs.permute(0, 2, 3, 4, 1)
        outputs = outputs.reshape(outputs.size(0), self.out_channels, *output_shape(inputs, self.kernel_size, self.padding, self.stride, self.dilation))

        return outputs


if __name__ == '__main__':

    net1 = FullAttenSUMNet(in_ch=1, out_ch=2).cuda()
    input = torch.randn(1,1,512,512)
    net2 = SUMNet(in_ch=1, out_ch=2).cuda()
    summary(net1, input_size=(1,256,256))
    summary(net2, input_size=(1,256,256))


    # out = net(input)
    # print(out.shape)