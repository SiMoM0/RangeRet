# main model RangeRet

import math
import torch
from torch import nn

from network.retnet import RetNet
from network.pyretnet import PyramidRetNet
from utils.vision_embedding import VisionEmbedding

from network.transformers import Transformers

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, prob=0.0):
        super(BasicConv2d, self).__init__()
        self.drop = nn.Dropout2d(p=prob)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.norm = nn.InstanceNorm2d(out_planes)
        self.bnorm = nn.BatchNorm2d(out_planes)
        self.gelu = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.drop(x)
        x = self.conv(x)
        x = self.bnorm(x)
        #x = self.norm(x)
        x = self.gelu(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, prob=0.0):
        super(MLP, self).__init__()
        self.drop = nn.Dropout(p=prob)
        self.mlp = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.bnorm = nn.BatchNorm2d(out_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.drop(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        x = self.bnorm(x)
        x = x.permute(0, 2, 3, 1)
        #x = self.norm(x)
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
        self.bnorm = nn.BatchNorm2d(in_dim)
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
        B, C, H, W = x.shape

        #x = self.dropout(x)
        
        x = self.bnorm(x)
        #x = self.inorm(x)
        x = self.convs(x)

        #x = self.norm(x)
        #x = self.mlp(x)

        #x = x.permute(0, 3, 1, 2)

        return x

class Embedding(nn.Module):
    def __init__(self, in_dim=5, out_dim=128, neighbors=16, H=64, W=1024):
        super(Embedding, self).__init__()
        self.H = H
        self.W = W
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neighbors = neighbors

        # input normalization
        self.norm = nn.BatchNorm1d(in_dim)

        # raw point embedding
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=1)

        # neighborhood embedding
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=False)
        )

        # merge points and neighborhood embeddings
        self.final = nn.Conv1d(2 * out_dim, out_dim, kernel_size=1, bias=True, padding=0)

    def forward(self, pc, neighbors, px, py):
        #print('Poitn cloud shape: ', pc.shape)
        #print('Neighbors shape: ', neighbors.shape)
        # transpose channels and number of points
        pc = pc.transpose(1, 2)
        neighbors = neighbors.transpose(1, 2)

        B, _, N = pc.shape

        #print('Transpose pc: ', pc.shape)
        #print('Transpose neighbors: ', neighbors.shape)

        pc = self.norm(pc)

        #print('point cloud normalized: ', pc.shape)

        # point embedding
        point_emb = self.conv1(pc)
        #print('point embedding: ', point_emb.shape)

        gather = []
        # gather neighbors around center point
        for ind_nn in range(1, neighbors.shape[1]): # remove first neighbors which is center point
            temp = neighbors[:, ind_nn:ind_nn+1, :].expand(-1, pc.shape[1], -1)
            #print('temp: ', temp.shape)
            gather.append(torch.gather(pc, 2, temp).unsqueeze(-1))

        # relative coordinates
        neigh_emb = torch.cat(gather, -1) - pc.unsqueeze(-1)
        #print('relative coordinates: ', neigh_emb.shape)
        # embedding
        neigh_emb = self.conv2(neigh_emb).max(-1)[0]
        #print('neighbors embeddings: ', neigh_emb.shape)

        feats = self.final(torch.cat([point_emb, neigh_emb], dim=1))
        #print('final feats: ', feats.shape)

        # create range image with features
        proj_feat = torch.full([self.H, self.W, self.out_dim], -1, dtype=torch.float32).to(feats.device)
        #print('proj_feat: ', proj_feat.shape)
        #print('px: ', px.shape)
        #print('py: ', py.shape)
        #print('feats: ', feats.shape)
        proj_feat[py, px] = feats.transpose(1, 2).squeeze(0)
        #print('proj_feat: ', proj_feat.shape)

        return proj_feat.permute(2, 0, 1).unsqueeze(0)


class SemanticHead(nn.Module):
    '''
    Semantic Head: two MLP layers to map feature dimension into number of classes
    '''
    def __init__(self, in_dim=[128, 128, 256], hidden_dim=128, height=64, width=1024, num_classes=20, dropout=0.0):
        super(SemanticHead, self).__init__()
        self.height = height
        self.width = width

        self.mlp1 = MLP(in_dim=in_dim[0], out_dim=hidden_dim)
        self.mlp2 = MLP(in_dim=in_dim[1], out_dim=hidden_dim)
        self.mlp3 = MLP(in_dim=in_dim[2], out_dim=hidden_dim)

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=hidden_dim),
            nn.GELU()
        )

        self.pred1 = nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1)
        self.pred2 = nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1)
        self.pred3 = nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1)

        self.final = nn.Linear(hidden_dim, num_classes)

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, rem):
        x1, x2, x3 = x

        x1 = self.mlp1(x1)
        # reshape to (B, C, H, W)
        x1 = x1.permute(0, 3, 1, 2)
        # bilinear interpolation
        x1 = torch.nn.functional.interpolate(x1, size=(self.height, self.width), mode='bilinear')

        x2 = self.mlp2(x2)
        x2 = x2.permute(0, 3, 1, 2)
        x2 = torch.nn.functional.interpolate(x2, size=(self.height, self.width), mode='bilinear')

        x3 = self.mlp3(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = torch.nn.functional.interpolate(x3, size=(self.height, self.width), mode='bilinear')

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.fuse(x)

        x1 = self.pred1(x1)
        x2 = self.pred2(x2)
        x3 = self.pred3(x3)

        # residual connection with REM output
        if rem is not None:
            x = x + rem

        x = x.permute(0, 2, 3, 1)

        x = self.final(x)

        # TODO use predictions from all stages ?
        return [x, x1, x2, x3]

class RangeRet(nn.Module):
    def __init__(self, model_params: dict, img_size=(64, 1024), activate_recurrent=False):
        super(RangeRet, self).__init__()
        self.H, self.W = img_size
        self.img_size = img_size
        self.patch_size = model_params['patch_size']
        self.stride = model_params['stride']
        self.in_dim = model_params['input_dims']
        self.rem_dim = model_params['rem_dim']
        self.decoder_dim = model_params['decoder_dim']

        # retnet params
        self.embed_dims = model_params['embed_dims']
        self.heads = model_params['heads']
        self.mlp_dim = model_params['mlp_dim']
        self.blocks = model_params['blocks']

        #self.patched_image = (math.floor((self.H - self.patch_size) / self.stride) + 1,
        #                      math.floor((self.W - self.patch_size) / self.stride) + 1)

        #print(f'Patched image size = {self.patched_image}')

        #self.rem = REM(self.in_dim, self.rem_dim, dropout=0.0)
        
        self.embedding = Embedding(self.in_dim, self.rem_dim, H=self.H, W=self.W)

        self.model = PyramidRetNet(img_size=self.img_size,
                                   patch_size=self.patch_size,
                                   strides=self.stride,
                                   in_dim=self.rem_dim,
                                   double_v_dim=True,
                                   embed_dims=self.embed_dims,
                                   heads=self.heads,
                                   mlp_dim=self.mlp_dim,
                                   blocks=self.blocks)
        
        self.head = SemanticHead([128, 128, 256], 128, self.H, self.W, 20)

        # transformers for ablation study
        #self.transformers = Transformers(self.layers, self.hidden_dim, self.ffn_size, self.num_head, self.patched_image)
    
    def forward(self, x, neighbors):
        # extract and prepare input
        pc = torch.cat([x[0], x[1].unsqueeze(-1), x[2].unsqueeze(-1)], dim=-1)
        px = x[3]
        py = x[4]

        # TODO for better performance dont use different vars
        rem_out = self.embedding(pc, neighbors, px, py)

        ret_out = self.model(rem_out)

        out = self.head(ret_out, rem_out)

        return out

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

    def recurrent(self, x):
        x = self.rem(x)

        x = self.viembed(x)

        incremental_state = {}
        outputs = []

        for i in range(x.shape[1]):
            o = self.retnet(x[:, i].unsqueeze(0), incremental_state)
            outputs.append(o)

        x = torch.cat(outputs, dim=1)

        x = torch.reshape(x, (x.shape[0], self.patched_image[0], self.patched_image[1], x.shape[2]))

        x = self.head(x)

        return x