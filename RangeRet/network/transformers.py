import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.xpos import XPOS

class Transformers(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, img_dim):
        super(Transformers, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.img_dim = img_dim
        self.slen = img_dim[0] * img_dim[1]

        self.attentions = nn.ModuleList([
            MultiHeadAttention(self.hidden_dim, self.heads) for _ in range(layers)
        ])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, ffn_size),
                nn.ReLU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])

        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(layers)
        ])

    def forward(self, x):

        for i in range(self.layers):
            y = self.layer_norms_1[i](x + self.attentions[i](x))
            x = self.layer_norms_2[i](y + self.ffns[i](y))

        # reshape
        x = torch.reshape(x, (x.shape[0], self.img_dim[0], self.img_dim[1], x.shape[2]))

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.v_dim = hidden_size * 2
        self.heads = heads
        assert hidden_size % heads == 0, "hidden_size must be divisible by heads"
        self.head_size = self.v_dim // heads
        self.key_dim = self.hidden_size // self.heads

        self.scaling = self.key_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.v_dim, bias=False)
        self.out_proj = nn.Linear(self.v_dim, hidden_size, bias=False)

        self.xpos = XPOS(self.key_dim)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x):
        bsz, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seq_len, self.heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.heads, self.key_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.heads, self.head_size).transpose(1, 2)
        q = q.reshape(bsz * self.heads, seq_len, self.key_dim)
        k = k.reshape(bsz * self.heads, seq_len, self.key_dim)
        v = v.reshape(bsz * self.heads, seq_len, self.head_size)

        q = self.xpos(q)
        k = self.xpos(k)

        att = torch.bmm(q, k.transpose(1, 2))

        att = F.softmax(att, dim=-1, dtype=torch.float32).type_as(att)
        att = att * self.scaling

        att = torch.bmm(att, v)
        att = att.transpose(0, 1).reshape(seq_len, bsz, self.v_dim).transpose(0, 1)

        att = self.out_proj(att)

        return att