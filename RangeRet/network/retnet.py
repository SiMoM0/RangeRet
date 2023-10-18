import torch
import torch.nn as nn

from network.retention import MultiScaleRetention

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
    
    def forward(self, x):
        """
        X: (batch_size, number of patches, number of features)
        """
        for i in range(self.layers):
            y = self.retentions[i](self.layer_norms_1[i](x)) + x

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