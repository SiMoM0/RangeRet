# main model RangeRet

import math
import torch
from torch import nn

from network.retnet import RetNet
from network.vision_embedding import VisionEmbedding
from network.stem import ConvStem

from network.vit import VisionTransformer

from timm.layers import DropPath, trunc_normal_

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, prob=0.0):
        super(BasicConv2d, self).__init__()
        self.drop = nn.Dropout2d(p=prob)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_planes)
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
        
        self.norm = nn.BatchNorm2d(in_dim)
        self.dropout = nn.Dropout2d(p=dropout)

        #self.norm = nn.LayerNorm(in_dim)

        #self.mlp = nn.Sequential(
        #    MLP(in_dim=in_dim, out_dim=32),
        #    MLP(in_dim=32, out_dim=64),
        #    MLP(in_dim=64, out_dim=out_dim)
        #)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        '''
        x: (H, W, in_dim) range image
        '''

        #x = self.dropout(x)
        x = self.norm(x)
        x = self.convs(x)

        #x = self.norm(x)
        #x = self.mlp(x)

        #x = x.permute(0, 3, 1, 2)

        return x

class SemanticHead(nn.Module):
    '''
    Semantic Head: two MLP layers to map feature dimension into number of classes
    '''
    def __init__(self, in_dim, hidden_dim, height, width, patched_img, num_classes, dropout=0.0):
        super(SemanticHead, self).__init__()
        self.height = height
        self.width = width
        self.patched_img = patched_img

        self.mlp1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(hidden_dim, num_classes)

        self.norm = nn.BatchNorm2d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, rem):
        # reshape to 2d from (B, N, C) to (B, H, W, C)
        x = torch.reshape(x, (x.shape[0], self.patched_img[0], self.patched_img[1], x.shape[2]))
        # reshape to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # bilinear interpolation
        x = torch.nn.functional.interpolate(x, size=(self.height, self.width), mode='bilinear')

        # residual connection with REM output
        if rem is not None:
            x = x + rem

        # reshape to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        #x = self.dropout(x)
        x = self.mlp1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        #x = self.dropout(x)
        x = self.gelu(x)
        x = self.mlp2(x)

        return x
    
class Decoder(nn.Module):
    '''
    Head inspired by RangeViT: https://arxiv.org/pdf/2301.10222
    '''
    def __init__(self, in_dim, hidden_dim, height, width, patched_img, num_classes, dropout=0.0):
        super(Decoder, self).__init__()
        self.height = height
        self.width = width
        self.patched_img = patched_img

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hidden_dim),
        )

        self.out = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x, rem):
        # reshape to 2d from (B, N, C) to (B, H, W, C)
        x = torch.reshape(x, (x.shape[0], self.patched_img[0], self.patched_img[1], x.shape[2]))
        # reshape to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # bilinear interpolation
        x = torch.nn.functional.interpolate(x, size=(self.height, self.width), mode='bilinear')

        # residual connection with REM output
        if rem is not None:
            x = x + rem
        
        x = self.conv1(x)
        x = self.out(x)
        x = x.permute(0, 2, 3, 1) # (B, H, W, C)
        
        return x

class RangeRet(nn.Module):
    def __init__(self, model_params: dict, resolution, num_classes=20, activate_recurrent=False):
        super(RangeRet, self).__init__()
        self.H = resolution[0]
        self.W = resolution[1]
        self.patch_size = model_params['patch_size']    # tuple (Hp, Wp)
        self.stride = model_params['stride']
        self.pool = model_params['pool']

        assert isinstance(self.patch_size, (list, tuple)), 'patch_size must be a tuple or list'
        assert isinstance(self.stride, (list, tuple)), 'stride must be a tuple or list'

        self.in_dim = model_params['input_dim']
        self.base_dim = model_params['base_dim']
        self.dim = model_params['dim']
        self.decoder_dim = model_params['decoder_dim']
        self.drop_path_rate = model_params['drop']
        self.num_classes = num_classes

        # backbone type
        self.bb = model_params['backbone']

        # retnet parameters
        self.layers = model_params['retnet']['layers']
        self.model_dim = model_params['retnet']['model_dim']
        self.mlp_ratio = model_params['retnet']['mlp_ratio']
        self.num_head = model_params['retnet']['num_head']
        self.double_v_dim = model_params['retnet']['double_v_dim']

        assert self.dim == self.model_dim, 'conv stem dim must be equal to model dim'

        self.patched_image = (math.floor((self.H - self.patch_size[0]) / self.stride[0]) + 1,
                              math.floor((self.W - self.patch_size[1]) / self.stride[1]) + 1)

        print(f'Patched image size = {self.patched_image}')

        #self.rem = REM(self.in_dim, self.dim, dropout=0.0)
        self.rem = ConvStem(in_channels=self.in_dim, base_channels=self.base_dim, img_size=(self.H, self.W), patch_stride=(self.patch_size, self.patch_size), embed_dim=self.dim, flatten=False, hidden_dim=self.dim)
        
        self.viembed = VisionEmbedding(self.H, self.W, self.patch_size, self.dim, self.model_dim, self.stride, self.pool) # H, W, patch size, input channel, output features
        
        if self.bb == 'retnet':
            self.backbone = RetNet(self.layers, self.model_dim, self.mlp_ratio, self.num_head, self.patched_image, self.double_v_dim, self.drop_path_rate, activate_recurrent=activate_recurrent) #layers=4, hidden_dim=128, ffn_size=256, num_head=4, (patched_image_h, patched_image_w), v_dim=double
        elif self.bb == 'vit':
            self.backbone = VisionTransformer(self.patched_image, self.model_dim, self.layers, self.num_head, self.mlp_ratio, drop_path_rate=self.drop_path_rate)
        
        self.head = SemanticHead(self.model_dim, self.decoder_dim, self.H, self.W, self.patched_image, self.num_classes)
        #self.head = Decoder(self.model_dim, self.decoder_dim, self.H, self.W, self.patched_image, self.num_classes)
    
    def forward(self, x):
        x = self.rem(x)

        residual = x
        
        x = self.viembed(x)

        x = self.backbone(x)

        x = self.head(x, residual)

        return x