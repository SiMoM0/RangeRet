import math
import torch
import torch.nn as nn

#from network.retention import MultiScaleRetention
from network.msr import MultiScaleRetention

class RetNet(nn.Module):
    '''
    RetNet block

    * layers: number of blocks
    * hidden_dim: input feature
    * ffn_size: dimension of feed-forward network
    * heads: number of heads
    * img_dim: shape of input image
    '''
    def __init__(self, layers, hidden_dim, ffn_size, heads, img_dim, double_v_dim=True):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.img_dim = img_dim
        self.slen = img_dim[0] * img_dim[1]
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()
        #self.D = [self._get_D(img_dim[0] * img_dim[1], g).cuda() for g in self.gammas]
        self.retnet_rel_pos = self.get_rel_pos()

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim, img_dim[0] * img_dim[1])
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                #nn.Dropout(p=0.15),
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                #nn.Dropout(p=0.15),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
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

    def get_rel_pos(self):
        angle = 1.0 / (10000 ** torch.linspace(0, 1, self.hidden_dim // self.heads // 2)).cuda()
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        decay = torch.log(1 - 2 ** (-5 - torch.arange(self.heads, dtype=torch.float))).cuda()

        index = torch.arange(self.slen).to(decay)
        sin = torch.sin(index[:, None] * angle[None, :])
        cos = torch.cos(index[:, None] * angle[None, :])
        mask = torch.tril(torch.ones(self.slen, self.slen).to(decay))
        mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
        mask = torch.exp(mask * decay[:, None, None])
        mask = torch.nan_to_num(mask)
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
        retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

    def forward(self, x):
        """
        X: (batch_size, number of patches, number of features)
        """
        for i in range(self.layers):
            y = self.retentions[i](self.layer_norms_1[i](x), self.retnet_rel_pos) + x

            x = self.ffns[i](self.layer_norms_2[i](y)) + y

        # reshape to patched image shape
        x = torch.reshape(x, (x.shape[0], self.img_dim[0], self.img_dim[1], x.shape[2]))

        return x

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
        
        return x_n, s_ns
    
    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i
        
        return x_i, r_is