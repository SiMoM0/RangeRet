import math
import torch
import torch.nn as nn

from network.msr import MultiScaleRetention

from timm.layers import DropPath, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class GLU(nn.Module):
    def __init__(
        self, embed_dim, ffn_dim, act=nn.GELU, dropout=0.0, activation_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.activation_fn = act()
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim, bias=False)
        self.gate = nn.Linear(self.embed_dim, ffn_dim, bias=False)

    def forward(self, x, H, W):
        x_shape = x.shape
        x = x.reshape(-1, x.size(-1))
        g = self.gate(x)
        x = self.fc1(x)
        x = self.activation_fn(x.float()).type_as(x) * g
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = x.view(x_shape)
        x = self.dropout_module(x)
        return x

class RetNet(nn.Module):
    '''
    RetNet block

    * layers: number of blocks
    * hidden_dim: input feature
    * mlp_ratio: dimension of feed-forward network
    * heads: number of heads
    * img_dim: shape of input image
    '''
    def __init__(self, layers, hidden_dim, mlp_ratio, heads, img_dim, double_v_dim=True, drop_path_rate=0.0, activate_recurrent=False):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = int(hidden_dim * mlp_ratio)
        self.heads = heads
        self.img_dim = img_dim
        self.slen = img_dim[0] * img_dim[1]
        self.activate_recurrent = activate_recurrent
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()
        #self.D = [self._get_D(img_dim[0] * img_dim[1], g).cuda() for g in self.gammas]
        self.retnet_rel_pos = self.get_rel_pos(self.activate_recurrent, manhattan=True)

        self.retentions = nn.ModuleList([
            MultiScaleRetention(self.hidden_dim, self.heads, double_v_dim, self.slen)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            Mlp(self.hidden_dim, self.mlp_dim, self.hidden_dim)
            #GLU(self.hidden_dim, self.mlp_dim, act=nn.GELU)
            for _ in range(layers)
        ])
        self.norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            #DyT(hidden_dim)
            for _ in range(layers)
        ])
        self.norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            #DyT(hidden_dim)
            for _ in range(layers)
        ])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.drop_path = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity()
            for i in range(layers)
        ])

    def _get_D(self, sequence_length, gamma):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        # normalization
        D = D / D.sum(dim=-1, keepdim=True).sqrt()

        return D

    def get_rel_pos(self, activate_recurrent=False, manhattan=True):
        angle = 1.0 / (10000 ** torch.linspace(0, 1, self.hidden_dim // self.heads // 2)).cuda()
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(self.heads, dtype=torch.float))).cuda()
        # alternative decay described in the paper
        #gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), self.heads))).cuda()


        if activate_recurrent:
            print('Using recurrent relative position encoding')
            sin = torch.sin(angle * (self.slen - 1))
            cos = torch.cos(angle * (self.slen - 1))
            retention_rel_pos = ((sin, cos), decay.exp())
        elif manhattan:
            print('Using Manhattan relative position encoding')
            index = torch.arange(self.slen).to(decay)
            sin = torch.sin(index[:, None] * angle[None, :])
            cos = torch.cos(index[:, None] * angle[None, :])
            rows = torch.arange(self.slen).to(decay) // self.img_dim[1]
            cols = torch.arange(self.slen).to(decay) % self.img_dim[1]
            row_diff = torch.abs(rows[:, None] - rows)
            col_diff = torch.abs(cols[:, None] - cols)
            col_diff = torch.minimum(col_diff, self.img_dim[1] - col_diff)  # circular distance
            mask = row_diff + col_diff
            # exponential mapping
            mask = (mask / mask.max())** 2 * (self.slen - 1)
            # mask = ((mask - 1) / (self.img_dim[0] + self.img_dim[1] - 3))** 2 * (self.slen - 1) + 1
            mask.fill_diagonal_(0)
            # for softmax
            mask = mask * decay[:, None, None]
            #comment next two lines for softmax
            # mask = torch.exp(mask * decay[:, None, None])
            # mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)
        else:
            print('Using Euclidean relative position encoding')
            index = torch.arange(self.slen).to(decay)
            sin = torch.sin(index[:, None] * angle[None, :])
            cos = torch.cos(index[:, None] * angle[None, :])
            mask = torch.tril(torch.ones(self.slen, self.slen).to(decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * decay[:, None, None])
            mask = torch.nan_to_num(mask)
            # create upper triangle
            mask = mask + mask.transpose(1, 2)
            mask = mask - torch.eye(self.slen).to(decay)
            # normalization
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

    def forward(self, x, incremental_state=None):
        """
        X: (batch_size, number of patches, number of features)
        """

        is_first_step = self.is_first_step(incremental_state)
    
        for i in range(self.layers):
            if incremental_state is None or is_first_step:
                if is_first_step and incremental_state is not None:
                    if i not in incremental_state:
                        incremental_state[i] = {}
            else:
                if i not in incremental_state:
                    incremental_state[i] = {}

            y = self.drop_path[i](self.retentions[i](self.norms1[i](x), self.retnet_rel_pos, incremental_state)) + x
            #y = self.retentions[i](self.norms1[i](x), self.D) + x

            x = self.drop_path[i](self.ffns[i](self.norms2[i](y), self.img_dim[0], self.img_dim[1])) + y

        # reshape to patched image shape
        # if incremental_state is None:
        #     x = torch.reshape(x, (x.shape[0], self.img_dim[0], self.img_dim[1], x.shape[2]))

        return x

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.norms1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.norms2[i](y_n)) + y_n
        
        return x_n, s_ns
    
    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.norms1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.norms2[j](y_i)) + y_i
        
        return x_i, r_is

    def is_first_step(self, incremental_state):
        if incremental_state is None:
            return False
        return incremental_state.get("is_first_step", False)

class DWConv(nn.Module):
    def __init__(self, dim=128):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DyT(nn.Module):
    '''
    Dynamic Tanh from the paper "Transformers without Normalization" (https://arxiv.org/pdf/2503.10622)
    '''
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias