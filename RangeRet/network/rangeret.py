# main model RangeRet

import math
import torch
from torch import nn

from network.retnet import RetNet
from utils.vision_embedding import VisionEmbedding

def compute_shape(H, W, patch_size, stride):
    '''
    Compute patched image shape as a tuple (new_H, new_W)
    '''
    return (math.floor((H - patch_size) / stride) + 1,
            math.floor((W - patch_size) / stride) + 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = nn.InstanceNorm2d(out_planes)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x

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

        self.mlp1 = nn.Linear(in_dim, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, out_dim)

        self.gelu = nn.GELU()

        self.norm = nn.LayerNorm(in_dim)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(128)
        self.norm3 = nn.LayerNorm(out_dim)

        self.convs = nn.Sequential(BasicConv2d(5, 32, kernel_size=3, padding=1),
                                BasicConv2d(32, 64, kernel_size=3, padding=1),
                                BasicConv2d(64, 128, kernel_size=3, padding=1))
        
        self.inorm = nn.InstanceNorm2d(in_dim)

    def forward(self, x):
        '''
        x: (H, W, in_dim) range image
        '''
        # TODO normalize data ?
        x = x.permute(0, 3, 1, 2) # for conv2d REM (B, C, H, W)
        x = self.inorm(x)
        x = self.convs(x)

        #x = self.norm(x)
        #x = self.mlp1(x)
        #x = self.norm1(x)
        #x = self.gelu(x)
        #x = self.mlp2(x)
        #x = self.norm2(x)
        #x = self.gelu(x)
        #x = self.mlp3(x)
        #x = self.norm3(x)
        #x = self.gelu(x)
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
        self.linears = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in in_dim
        ])

        self.gelu = nn.GELU()

        self.linear_fuse = nn.Sequential(
            nn.Linear(hidden_dim*len(self.linears), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

        # semantic heads
        self.sem_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(len(self.linears))
        ])

        #self.softmax = nn.Softmax(-1)

        #self.deconv = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        #self.deconv2 = nn.ConvTranspose2d(in_dim, in_dim, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))

        #self.conv1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        #self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        # feature maps as the number of stages
        feat_maps = []

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
            feat_maps.append(feat.permute(0, 2, 3, 1))

        # fuse feature and feed to final MLPs
        x = self.linear_fuse(torch.cat(feat_maps, dim=3))

        # semantic head of each feature maps
        outs = []

        for i in range(len(feat_maps)):
            outs.append(self.sem_heads[i](feat_maps[i]))


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

        return x, outs

class Stage(nn.Module):
    '''
    Stage class containing Vision Embedding module and RetNet architecture
        index = index of stage (starting from 0)
    '''
    def __init__(self, index, model_params: dict):
        super(Stage, self).__init__()
        # input
        self.index = index
        self.H = model_params['H']
        self.W = model_params['W']
        self.patch_size = model_params['patch_size'][index]
        self.stride = model_params['stride'][index]
        self.rem_dim = model_params['rem_dim']
        # retnet parameters
        self.layers = model_params['retnet']['layers'][index]
        self.hidden_dim = model_params['retnet']['hidden_dim'][index]
        self.prev_hidden_dim = model_params['retnet']['hidden_dim'][index-1] if index > 0 else self.rem_dim
        self.ffn_size = model_params['retnet']['ffn_size'][index]
        self.num_head = model_params['retnet']['num_head'][index]
        self.double_v_dim = model_params['retnet']['double_v_dim']
        self.downsampling = model_params['downsampling'][index]
        self.next_downsampling = model_params['downsampling'][index+1] if index+1 < len(model_params['downsampling']) else self.downsampling

        # input H, W
        self.inputH = self.H // self.downsampling
        self.inputW = self.W // self.downsampling

        # patched image H, W
        self.patched_image = compute_shape(self.inputH, self.inputW, self.patch_size, self.stride)
        print(f'Patched image size stage {index} = {self.patched_image}')

        # input H, W for next block
        self.outputH = self.H // self.next_downsampling
        self.outputW = self.W // self.next_downsampling

        # modules
        self.viembed = VisionEmbedding(self.inputH, self.inputW, self.patch_size, self.prev_hidden_dim, self.hidden_dim, self.stride) # H, W, patch size, input channel, output features
        self.retnet = RetNet(self.layers, self.hidden_dim, self.ffn_size, self.num_head, self.patched_image, self.double_v_dim) #ex layers=4, hidden_dim=128, ffn_size=256, num_head=4, (patched_image_h, patched_image_w), v_dim=double


    def forward(self, x):
        #print(f'Stage {self.index} : input shape {x.shape}')

        x = self.viembed(x)
        #print(f'Stage {self.index} : embedding shape {x.shape}')

        x = self.retnet(x)
        #print(f'Stage {self.index} : ret shape {x.shape}')

        # input for next stage
        out = x.permute(0, 3, 1, 2)
        #print(f'Stage {self.index} : out shape {out.shape}')
        out = torch.nn.functional.interpolate(out, size=(self.outputH, self.outputW), mode='bilinear')
        #print(f'Stage {self.index} : out inter shape {out.shape}')
        return x, out

class RangeRet(nn.Module):
    def __init__(self, model_params: dict):
        super(RangeRet, self).__init__()
        self.H = model_params['H']
        self.W = model_params['W']
        self.in_dim = model_params['input_dims']
        self.rem_dim = model_params['rem_dim']
        self.decoder_dim = model_params['decoder_dim']
        self.num_stage = model_params['stages']
        self.hidden_dims = model_params['retnet']['hidden_dim']

        # model
        self.rem = REM(self.in_dim, self.rem_dim)

        self.stages = nn.ModuleList([
            Stage(i, model_params) for i in range(self.num_stage)
        ])

        self.head = SemanticHead(self.hidden_dims, self.decoder_dim, self.H, self.W, 20)
    
    def forward(self, x):
        outs = []

        # TODO for better performance dont use different vars
        x = self.rem(x)
        #print(rem_out.shape) # (1, 64, 1024, 128)

        # reshape to (B, C, H, W)
        #x = x.permute(0, 3, 1, 2) # (1, 128, 64, 1024)

        for i in range(self.num_stage):
            # return feature learned in the stage and input for next stage
            feat, x = self.stages[i](x)
            outs.append(feat)

        # return predicted range image and intermediate predictions 
        out, segs = self.head(outs)

        return out, segs