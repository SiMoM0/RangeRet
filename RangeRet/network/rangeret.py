# main model RangeRet

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
        #self.bnorm = nn.BatchNorm1d(out_dim)

        self.gelu = nn.GELU()

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
        # TODO add some batch normalization or dropout ?

        #out = self.bnorm(x3)

        return x

class SemanticHead(nn.Module):
    '''
    Semantic Head: two MLP layers to map feature dimension into number of classes
    '''
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(SemanticHead, self).__init__()
        self.mlp1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(hidden_dim, num_classes)
        #self.softmax = nn.Softmax(-1)

    def forward(self, x):
        # reshape to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # bilinear interpolation
        x = torch.nn.functional.interpolate(x, size=(64, 1024), mode='bilinear')
        # reshape to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        x = self.mlp1(x)
        x = self.gelu(x)
        x = self.mlp2(x)
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

        self.patched_image = (round((self.H - self.patch_size) / self.stride) + 1,
                              round((self.W - self.patch_size) / self.stride) + 1)

        print(self.patched_image)

        self.rem = REM(self.in_dim, self.rem_dim)
        self.viembed = VisionEmbedding(self.H, self.W, self.patch_size, self.rem_dim, self.rem_dim, self.stride) # H, W, patch size, input channel, output features
        # TODO add 4 stages of RetNet with different downsampling
        self.retnet = RetNet(self.layers, self.hidden_dim, self.ffn_size, self.num_head, self.patched_image, self.double_v_dim) #layers=4, hidden_dim=128, ffn_size=256, num_head=4, patched_image_h, patched_image_w, v_dim=double
        # TODO set 4 decoders as the number of stages for downsampling
        self.head = SemanticHead(self.rem_dim, self.decoder_dim, 20)
    
    def forward(self, x):
        # TODO for better performance dont use different vars
        rem_out = self.rem(x)

        # reshape to (B, C, H, W)
        rem_out = rem_out.permute(0, 3, 1, 2)

        patches = self.viembed(rem_out)

        ret_out = self.retnet(patches)

        out = self.head(ret_out)

        return out