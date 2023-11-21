# main model RangeRet

import math
import torch
from torch import nn

from network.retnet import RetNet
from utils.vision_embedding import VisionEmbedding

from network.transformers import Transformers

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, prob=0.0):
        super(BasicConv2d, self).__init__()
        self.drop = nn.Dropout2d(p=prob)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = nn.InstanceNorm2d(out_planes)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.drop(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, prob=0.0):
        super(MLP, self).__init__()
        self.drop = nn.Dropout(p=prob)
        self.mlp = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.drop(x)
        x = self.mlp(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x

class REM(nn.Module):
    '''
    Range Embedding Module: map each point in the range image to a
    higher-dim embedding (128) using 3 BasicConv2d layers
    '''
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super(REM, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.convs = nn.Sequential(BasicConv2d(in_dim, 32, kernel_size=3, padding=1, prob=dropout),
                                BasicConv2d(32, 64, kernel_size=3, padding=1, prob=dropout),
                                BasicConv2d(64, out_dim, kernel_size=3, padding=1, prob=dropout))
        
        self.inorm = nn.InstanceNorm2d(in_dim)
        self.dropout = nn.Dropout2d(p=dropout)

        #self.norm = nn.LayerNorm(in_dim)

        #self.mlp = nn.Sequential(
        #    MLP(in_dim=in_dim, out_dim=32),
        #    MLP(in_dim=32, out_dim=64),
        #    MLP(in_dim=64, out_dim=out_dim)
        #)

    def forward(self, x):
        '''
        x: (H, W, in_dim) range image
        '''

        #x = self.dropout(x)
        x = self.inorm(x)
        x = self.convs(x)

        #x = self.norm(x)
        #x = self.mlp(x)

        #x = x.permute(0, 3, 1, 2)

        return x

class SemanticHead(nn.Module):
    '''
    Semantic Head: two MLP layers to map feature dimension into number of classes
    '''
    def __init__(self, in_dim, hidden_dim, height, width, num_classes, dropout=0.0):
        super(SemanticHead, self).__init__()
        self.height = height
        self.width = width

        self.mlp1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(hidden_dim, num_classes)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # reshape to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # bilinear interpolation
        # TODO refactor if batch size is greater than 1
        x = torch.nn.functional.interpolate(x, size=(self.height, self.width), mode='bilinear')

        # residual connection with REM output
        #x = x + rem

        # reshape to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        #x = self.dropout(x)
        x = self.mlp1(x)
        x = self.norm(x)
        #x = self.dropout(x)
        x = self.gelu(x)
        x = self.mlp2(x)

        return x

class RangeRet(nn.Module):
    def __init__(self, model_params: dict, activate_recurrent=False):
        super(RangeRet, self).__init__()
        self.H = model_params['H']
        self.W = model_params['W']
        self.patch_size = model_params['patch_size']
        self.stride = model_params['stride']
        self.in_dim = model_params['input_dims']
        self.rem_dim = model_params['rem_dim']
        self.decoder_dim = model_params['decoder_dim']

        # retnet parameters
        self.layers = model_params['retnet']['layers']
        self.hidden_dim = model_params['retnet']['hidden_dim']
        self.ffn_size = model_params['retnet']['ffn_size']
        self.num_head = model_params['retnet']['num_head']
        self.double_v_dim = model_params['retnet']['double_v_dim']

        self.patched_image = (math.floor((self.H - self.patch_size) / self.stride) + 1,
                              math.floor((self.W - self.patch_size) / self.stride) + 1)

        print(f'Patched image size = {self.patched_image}')

        self.rem = REM(self.in_dim, self.rem_dim, dropout=0.0)
        
        self.viembed = VisionEmbedding(self.H, self.W, self.patch_size, self.rem_dim, self.rem_dim, self.stride) # H, W, patch size, input channel, output features
        # TODO add 4 stages of RetNet with different downsampling
        self.retnet = RetNet(self.layers, self.hidden_dim, self.ffn_size, self.num_head, self.patched_image, self.double_v_dim, activate_recurrent) #layers=4, hidden_dim=128, ffn_size=256, num_head=4, (patched_image_h, patched_image_w), v_dim=double
        # TODO set 4 decoders as the number of stages for downsampling
        self.head = SemanticHead(self.rem_dim, self.decoder_dim, self.H, self.W, 20)

        # transformers for ablation study
        #self.transformers = Transformers(self.layers, self.hidden_dim, self.ffn_size, self.num_head, self.patched_image)
    
    def forward(self, x, context):
        # TODO for better performance dont use different vars
        rem_out = self.rem(x)

        patches = self.viembed(rem_out)

        ret_out = self.retnet(patches, context)

        #ret_out = self.transformers(patches)

        out = self.head(ret_out)

        return out, ret_out

    def forward_recurrent(self, x):
        x = self.rem(x)

        x = self.viembed(x)
        #print('patches: ', x.shape)

        s0 = torch.zeros(self.layers, self.num_head, 1, self.hidden_dim // self.num_head, self.hidden_dim // self.num_head * 2).cuda()

        xs = []

        for i in range(x.shape[1]):
            #print(i)
            xn, sn = self.retnet.forward_recurrent(x[:, i].unsqueeze(0), s0, i)
            xs.append(xn)
            s0 = sn

        x = torch.cat(xs, dim=1)
        #print(x.size())

        x = torch.reshape(x, (x.shape[0], self.patched_image[0], self.patched_image[1], x.shape[2]))

        x = self.head(x)

        return x

    def recurrent(self, x, incremental_state):
        x = self.rem(x)

        x = self.viembed(x)

        #for i in range(x.shape[1]):
        #    o = self.retnet(x[:, i].unsqueeze(0), incremental_state)
        #    outputs.append(o)

        #x = torch.cat(outputs, dim=1)

        x = self.retnet(x, incremental_state)

        x = torch.reshape(x, (x.shape[0], self.patched_image[0], self.patched_image[1], x.shape[2]))

        x = self.head(x)

        return x