# Pyramid Retentive Network as PVT

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.vision_embedding import VisionEmbedding
#from network.retention import MultiScaleRetention
from network.msr import MultiScaleRetention

def compute_slen(H, W, patch_size, stride):
    '''
    Compute sequence length of an image given its shape and convolution parameters
    '''
    return (math.floor((H - patch_size) / stride) + 1) * (math.floor((W - patch_size) / stride) + 1)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        
        return x

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, ratio=1):
        super().__init__()

        self.ret = MultiScaleRetention(dim, heads, double_v_dim=True)
        self.mlp = MLP(dim, mlp_dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, rel_pos):
        x = x + self.ret(self.norm1(x), rel_pos)
        x = x + self.mlp(self.norm2(x))

        return x

class PyramidRetNet(nn.Module):
    def __init__(self, img_size, patch_size=[7, 5, 3], strides=[4, 3, 2], in_dim=128, double_v_dim=True, embed_dims=[128, 128, 256],
                 heads=[4, 4, 4], mlp_dim=[256, 256, 512], blocks=[3, 4, 6], ratio=[16, 8, 2, 1]):
        super().__init__()

        self.viembed1 = VisionEmbedding(H=img_size[0],
                                       W=img_size[1],
                                       patch_size=patch_size[0],
                                       in_chans=in_dim,
                                       embed_dim=embed_dims[0],
                                       stride=strides[0])
        self.viembed2 = VisionEmbedding(H=img_size[0] // 2,
                                       W=img_size[1] // 2,
                                       patch_size=patch_size[1],
                                       in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1],
                                       stride=strides[1])
        self.viembed3 = VisionEmbedding(H=img_size[0] // 4,
                                       W=img_size[1] // 4,
                                       patch_size=patch_size[2],
                                       in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2],
                                       stride=strides[2])
        #self.viembed4 = VisionEmbedding(H=img_size[0] // 4,
        #                               W=img_size[1] // 4,
        #                               patch_size=patch_size,
        #                               in_chans=embed_dims[2],
        #                               embed_dim=embed_dims[3],
        #                               stride=1)

        # TODO remove refactor code
        self.mask1 = self.get_rel_pos(head_dim=embed_dims[0], num_head=heads[0], slen=3825, img_dim=(15, 255))
        self.mask2 = self.get_rel_pos(head_dim=embed_dims[1], num_head=heads[1], slen=1700, img_dim=(10, 170))
        self.mask3 = self.get_rel_pos(head_dim=embed_dims[2], num_head=heads[2], slen=889, img_dim=(7, 127))

        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], heads=heads[0], mlp_dim=mlp_dim[0], ratio=ratio[0]
            ) for _ in range(blocks[0])])
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], heads=heads[1], mlp_dim=mlp_dim[1], ratio=ratio[1]
            ) for _ in range(blocks[1])])
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], heads=heads[2], mlp_dim=mlp_dim[2], ratio=ratio[2]
            ) for _ in range(blocks[2])])
        #self.block4 = nn.ModuleList([Block(
        #    dim=embed_dims[3], heads=heads[3], mlp_dim=mlp_dim[3], ratio=ratio[3]
        #    ) for _ in range(blocks[3])])

        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        #self.norm4 = nn.LayerNorm(embed_dims[3])

    def get_rel_pos(self, head_dim, num_head, slen, img_dim, activate_recurrent=False, manhattan=True):
        angle = 1.0 / (10000 ** torch.linspace(0, 1, head_dim // num_head // 2)).to('cuda' if torch.cuda.is_available() else 'cpu')
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(num_head, dtype=torch.float))).to('cuda' if torch.cuda.is_available() else 'cpu')
        # alternative decay described in the paper
        #gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), self.heads))).cuda()


        if activate_recurrent:
            sin = torch.sin(angle * (slen - 1))
            cos = torch.cos(angle * (slen - 1))
            retention_rel_pos = ((sin, cos), decay.exp())
        elif manhattan:
            index = torch.arange(slen).to(decay)
            sin = torch.sin(index[:, None] * angle[None, :])
            cos = torch.cos(index[:, None] * angle[None, :])
            rows = torch.arange(slen).to(decay) // img_dim[1]
            cols = torch.arange(slen).to(decay) % img_dim[1]
            row_diff = torch.abs(rows[:, None] - rows)
            col_diff = torch.abs(cols[:, None] - cols)
            mask = row_diff + col_diff
            # exponential mapping
            mask = ((mask - 1) / (img_dim[0] + img_dim[1] - 3))** 2 * (slen - 1) + 1
            mask.fill_diagonal_(0)
            mask = torch.exp(mask * decay[:, None, None])
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)
        else:
            index = torch.arange(slen).to(decay)
            sin = torch.sin(index[:, None] * angle[None, :])
            cos = torch.cos(index[:, None] * angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * decay[:, None, None])
            mask = torch.nan_to_num(mask)
            # create upper triangle
            mask = mask + mask.transpose(1, 2)
            mask = mask - torch.eye(slen).to(decay)
            # normalization
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # for residual connections
        _x = F.interpolate(x, size=(32, 512), mode='bilinear')
        _x2 = F.interpolate(x, size=(16, 256), mode='bilinear')

        x, H, W = self.viembed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, self.mask1)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(32, 512), mode='bilinear')
        x = x + _x
        outs.append(x.permute(0, 2, 3, 1))

        x, H, W = self.viembed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, self.mask2)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(16, 256), mode='bilinear')
        x = x + _x2
        outs.append(x.permute(0, 2, 3, 1))

        x, H, W = self.viembed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, self.mask3)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(16, 256), mode='bilinear')
        outs.append(x.permute(0, 2, 3, 1))

        #x, H, W = self.viembed4(x)
        #for i, blk in enumerate(self.block4):
        #    x = blk(x, H, W)
        #x = self.norm4(x)
        #x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        #outs.append(x.permute(0, 2, 3, 1))

        return outs