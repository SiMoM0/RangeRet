import math
import torch
from torch import nn

from utils.xpos import XPOS

class SimpleRetention(nn.Module):
    def __init__(self, hidden_size, gamma, head_size, double_v_dim, num_patches):
        '''
        Simple retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        '''
        super(SimpleRetention, self).__init__()

        self.hidden_size = hidden_size  # d_model
        self.head_size = head_size      # d_head

        self.v_dim = head_size * 2 if double_v_dim else head_size
        self.gamma = gamma

        self.W_Q = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size, requires_grad=True)
        self.W_K = nn.Parameter(torch.randn(hidden_size, head_size) / hidden_size, requires_grad=True)
        self.W_V = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size, requires_grad=True)

        #self.wq = nn.Linear(hidden_size, head_size, bias=False)
        #self.wk = nn.Linear(hidden_size, head_size, bias=False)
        #self.wv = nn.Linear(hidden_size, self.v_dim, bias=False)
        
        self.D = self._get_D(num_patches).cuda()

        self.xpos = XPOS(head_size)

        #elf.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wq.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.wk.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.wv.weight, gain=2 ** -2.5)

    def forward(self, x):
        '''
        Parallel (default) representation of the retention mechanism.\n
        ```x```: (batch_size, number of patches, number of features) ex: (B, H*W/(p**2), 128)
        '''
        #sequence_length = x.shape[1]
        #D = self._get_D(sequence_length).to(self.W_Q.device)

        Q = (x @ self.W_Q)
        K = (x @ self.W_K)

        #Q = self.wq(x)
        #K = self.wk(x)

        Q = self.xpos(Q)
        K = self.xpos(K, downscale=True)

        V = x @ self.W_V
        #V = self.wv(x)
        ret = (Q @ K.permute(0, 2, 1)) * self.D.unsqueeze(0)
        #ret = torch.matmul(Q, K.permute(0, 2, 1)) * self.D.unsqueeze(0)
        
        return ret @ V
        
    def forward_recurrent(self, x_n, s_n_1, n):
        """
        Recurrent representation of the retention mechanism.
        x_n: (batch_size, 1, hidden_size)
        s_n_1: (batch_size, hidden_size, v_dim)
        """

        Q = (x_n @ self.W_Q)
        K = (x_n @ self.W_K)

        Q = self.xpos(Q, n+1)
        K = self.xpos(K, n+1, downscale=True)

        V = x_n @ self.W_V

        # K: (batch_size, 1, hidden_size)
        # V: (batch_size, 1, v_dim)
        # s_n = gamma * s_n_1 + K^T @ V

        s_n = self.gamma * s_n_1 + (K.transpose(-1, -2) @ V)
        
        return (Q @ s_n), s_n
    
    def forward_chunkwise(self, x_i, r_i_1, i):
        """
        Chunkwise representation of the retention mechanism.
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1: (batch_size, hidden_size, v_dim)
        """
        batch, chunk_size, _ = x_i.shape
        D = self._get_D(chunk_size)

        Q = (x_i @ self.W_Q)
        K = (x_i @ self.W_K)

        Q = self.xpos(Q, i * chunk_size)
        K = self.xpos(K, i * chunk_size, downscale=True)

        V = x_i @ self.W_V
        
        r_i =(K.transpose(-1, -2) @ (V * D[-1].view(1, chunk_size, 1))) + (self.gamma ** chunk_size) * r_i_1

        inner_chunk = ((Q @ K.transpose(-1, -2)) * D.unsqueeze(0)) @ V
        
        #e[i,j] = gamma ** (i+1)
        e = torch.zeros(batch, chunk_size, 1)
        
        for _i in range(chunk_size):
            e[:, _i, :] = self.gamma ** (_i + 1)
        
        cross_chunk = (Q @ r_i_1) * e
        
        return inner_chunk + cross_chunk, r_i

    def _get_D(self, sequence_length):
        n = torch.arange(sequence_length).unsqueeze(1)
        m = torch.arange(sequence_length).unsqueeze(0)

        # Broadcast self.gamma ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.gamma ** (n - m)) * (n >= m).float()  #this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D[D != D] = 0

        return D

class MultiScaleRetention(nn.Module):
    def __init__(self, hidden_size, heads, double_v_dim, num_patches):
        """
        Multi-scale retention mechanism based on the paper
        "Retentive Network: A Successor to Transformer for Large Language Models"[https://arxiv.org/pdf/2307.08621.pdf]
        """
        super(MultiScaleRetention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2 if double_v_dim else hidden_size
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = hidden_size // heads
        self.head_v_dim = hidden_size * 2 if double_v_dim else hidden_size
        
        self.gammas = (1 - torch.exp(torch.linspace(math.log(1/32), math.log(1/512), heads))).detach().cpu().tolist()

        self.swish = lambda x: x * torch.sigmoid(x)
        self.W_G = nn.Parameter(torch.randn(hidden_size, self.v_dim) / hidden_size)
        self.W_O = nn.Parameter(torch.randn(self.v_dim, hidden_size) / hidden_size)
        #self.wg = nn.Linear(hidden_size, self.v_dim, bias=False)
        #self.wo = nn.Linear(self.v_dim, hidden_size, bias=False)
        self.group_norm = nn.GroupNorm(heads, self.v_dim)

        self.retentions = nn.ModuleList([
            SimpleRetention(self.hidden_size, gamma, self.head_size, double_v_dim, num_patches) for gamma in self.gammas
        ])

        #self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.wg.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.wo.weight)

    def forward(self, x):
        """
        parallel representation of the multi-scale retention mechanism
        """

        # apply each individual retention mechanism to X
        Y = []
        for i in range(self.heads):
            #print('single retention')
            Y.append(self.retentions[i](x))
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x @ self.W_G) * Y) @ self.W_O
        #return self.wo(self.swish(self.wg(x)))

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        recurrent representation of the multi-scale retention mechanism
        x_n: (batch_size, 1, hidden_size)
        s_n_1s: (batch_size, heads, head_size, head_size)

        """
    
        # apply each individual retention mechanism to a slice of X
        Y = []
        s_ns = []
        for i in range(self.heads):
            y, s_n = self.retentions[i].forward_recurrent(
                x_n[:, :, :], s_n_1s[i], n
                )
            Y.append(y)
            s_ns.append(s_n)
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)
        
        return (self.swish(x_n @ self.W_G) * Y) @ self.W_O, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        chunkwise representation of the multi-scale retention mechanism
        x_i: (batch_size, chunk_size, hidden_size)
        r_i_1s: (batch_size, heads, head_size, head_size)
        """
        batch, chunk_size, _ = x_i.shape

        # apply each individual retention mechanism to a slice of X
        Y = []
        r_is = []
        for j in range(self.heads):
            y, r_i = self.retentions[j].forward_chunkwise(
                x_i[:, :, :], r_i_1s[j], i
                )
            Y.append(y)
            r_is.append(r_i)
        
        
        Y = torch.cat(Y, dim=2)
        Y_shape = Y.shape
        Y = self.group_norm(Y.reshape(-1, self.v_dim)).reshape(Y_shape)

        return (self.swish(x_i @ self.W_G) * Y) @ self.W_O, r_is