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
        x1 = self.mlp1(x)
        x2 = self.gelu(x1)
        x3 = self.mlp2(x2)
        x4 = self.gelu(x3)
        x5 = self.mlp3(x4)
        out = self.gelu(x5)
        # TODO add some batch normalization or dropout ?

        #out = self.bnorm(x3)

        return out

class Decoder(nn.Module):
    '''
    Semantic Head: two MLP layers to map feature dimension into number of classes
    '''
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(Decoder, self).__init__()
        self.mlp1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        x1 = self.mlp1(x)
        x2 = self.gelu(x1)
        x3 = self.mlp2(x2)
        out = self.softmax(x3) # TODO use softmax or something else ?

        # TODO add dropout or batchnorm ?

        return out

class RangeRet(nn.Module):
    def __init__(self):
        super(RangeRet, self).__init__()
        self.rem = REM(5, 128)
        #self.viembed = VisionEmbedding(64, 1024, 4, 5, 128)
        # TODO add 4 stages of RetNet with different downsampling
        self.retnet = RetNet(4, 128, 256, 4, double_v_dim=True) #layers=4, hidden_dim=128, ffn_size=256, num_head=4, v_dim=double
        # TODO set 4 decoders as the number of stages for downsampling
        self.decoder = Decoder(128, 64, 20)
    
    def forward(self, x):
        rem_out = self.rem(x)
        #rem_out = self.viembed(x)

        #print(rem_out.shape)

        ret_out = self.retnet(rem_out)

        out = self.decoder(ret_out)

        return out