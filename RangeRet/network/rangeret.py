# main model RangeRet

import math
import torch
from torch import nn

from network.retnet import RetNet
from utils.vision_embedding import VisionEmbedding

# TODO unofficial RangeFormer implementation uses Conv2D
class REM(nn.Module):
    '''
    Range Embedding Module: map each point in the range image to a
    higher-dim embedding (128) using 3 MLP layers
    '''
    def __init__(self, in_dim, out_dim):
        super(REM, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.mlp1 = nn.Linear(in_dim, 32)
        self.mlp2 = nn.Linear(32, 64)
        self.mlp3 = nn.Linear(64, out_dim)

        self.gelu = nn.GELU()

        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        '''
        x: (H, W, in_dim) range image
        '''
        # TODO normalize data ?
        x = self.mlp1(x)
        x = self.gelu(x)
        x = self.mlp2(x)
        x = self.gelu(x)
        x = self.mlp3(x)
        x = self.gelu(x)
        x = self.norm(x)
        # TODO add some batch normalization or dropout ?

        return x

class SemanticHead(nn.Module):
    '''
    Semantic Head: two MLP layers to map feature dimension into number of classes
    '''
    def __init__(self, in_dim, hidden_dim, height, width, num_classes):
        super(SemanticHead, self).__init__()
        self.height = height
        self.width = width

        # channel unification
        self.linears = nn.ModuleList([nn.Linear(in_dim, hidden_dim),
                                      nn.Linear(in_dim, hidden_dim)])
        self.gelu = nn.GELU()

        self.linear_fuse = nn.Sequential(
            nn.Linear(hidden_dim*len(self.linears), hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
        #self.softmax = nn.Softmax(-1)

        #self.deconv = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #self.deconv2 = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        #self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        # channel unification
        heads = []

        for i in range(len(x)):
            #print(x[i].shape)
            # channel unification
            feat = self.linears[i](x[i])
            #print(feat.shape)
            feat = self.gelu(feat)
            #print(feat.shape)
            # reshape to (B, C, H, W)
            feat = feat.permute(0, 3, 1, 2)
            # bilinear interpolation
            feat = torch.nn.functional.interpolate(feat, size=(self.height, self.width), mode='bilinear', align_corners=False)
            #print(feat.shape) # (B, C, H, W)
            # reshape to (B, H, W, C)
            heads.append(feat.permute(0, 2, 3, 1))

        # fuse feature and feed to final MLPs
        x = self.linear_fuse(torch.cat(heads, dim=3))

        # reshape to (B, C, H, W)
        #x = x.permute(0, 3, 1, 2)
        # bilinear interpolation
        # TODO refactor if batch size is greater than 1
        #x = torch.nn.functional.interpolate(x, size=(self.height, self.width), mode='bilinear')
        
        # deconv approach
        #x = self.deconv(x)
        #print(x.shape)
        #x = self.deconv(x)
        #print(x.shape)
        
        #x = self.conv1(x)
        #x = self.conv2(x)

        # reshape to (B, H, W, C)
        #x = x.permute(0, 2, 3, 1)

        #x = self.mlp1(x)
        #x = self.gelu(x)
        #x = self.norm(x)
        #x = self.mlp2(x)
        #out = self.softmax(x3) # TODO use softmax only for NLL

        # TODO add dropout or batchnorm ?

        return x

class RangeRet(nn.Module):
    def __init__(self, model_params: dict):
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

        self.rem = REM(self.in_dim, self.rem_dim)
        self.viembed = VisionEmbedding(self.H, self.W, self.patch_size, self.rem_dim, self.rem_dim, self.stride) # H, W, patch size, input channel, output features
        # TODO add 4 stages of RetNet with different downsampling
        self.retnet1 = RetNet(self.layers, self.hidden_dim, self.ffn_size, self.num_head, self.patched_image, self.double_v_dim) #layers=4, hidden_dim=128, ffn_size=256, num_head=4, (patched_image_h, patched_image_w), v_dim=double
        
        self.viembed2 = VisionEmbedding(self.H // 2, self.W // 2, self.patch_size, self.rem_dim, self.rem_dim, self.stride)
        self.retnet2 = RetNet(self.layers, self.hidden_dim, self.ffn_size, self.num_head, (7, 127), double_v_dim=True)

        # TODO set 4 decoders as the number of stages for downsampling
        self.head = SemanticHead(self.rem_dim, self.decoder_dim, self.H, self.W, 20)
    
    def forward(self, x):

        outs = []

        # TODO for better performance dont use different vars
        rem_out = self.rem(x)
        #print(rem_out.shape) # (1, 64, 1024, 128)

        # reshape to (B, C, H, W)
        rem_out = rem_out.permute(0, 3, 1, 2) # (1, 128, 64, 1024)

        patches = self.viembed(rem_out)
        #print(patches.shape) # (1, 3825, 128)

        ret_out = self.retnet1(patches)
        outs.append(ret_out)
        #print(ret_out.shape) # (1, 15, 255, 128)
        
        ret_out = ret_out.permute(0, 3, 1, 2)
        #print(ret_out.shape) # (1, 128, 15, 255)
        ret_out = torch.nn.functional.interpolate(ret_out, size=(32, 512), mode='bilinear')
        #print(ret_out.shape) # (1, 128, 32, 512)

        # using REM output
        patches = self.viembed2(ret_out)
        #print(patches.shape) # (1, 889, 128)

        ret_out = self.retnet2(patches)
        outs.append(ret_out)
        #print(ret_out.shape) # (1, 7, 127, 128)

        out = self.head(outs)

        return out