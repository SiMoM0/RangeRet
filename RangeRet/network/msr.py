# Multi-Scale Retention based on https://github.com/microsoft/torchscale/blob/main/torchscale/component/multiscale_retention.py

import math
import torch
from torch import nn

def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\

def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)

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
        self.head_size = self.v_dim // heads
        self.key_dim = self.hidden_size // self.heads
        
        self.scaling = self.key_dim ** -0.5

        self.swish = lambda x: x * torch.sigmoid(x)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size,self. hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.v_dim, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, self.v_dim, bias=False)
        self.out_proj = nn.Linear(self.v_dim, hidden_size, bias=False)

        #self.group_norm = nn.GroupNorm(self.heads, self.v_dim)
        self.group_norm = RMSNorm(self.head_size, eps=1e-5, elementwise_affine=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2 ** -2.5)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, seq_len, embed_dim = v.size()

        vr = v.view(bsz, seq_len, self.heads, self.head_size).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2) # bsz * m * seq_len * seq_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().sum(dim=-1, keepdim=True).abs().clamp(min=1)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(self, qr, kr, v, decay, incremental_state):
        bsz = v.size(0)

        v = v.view(bsz, self.heads, self.head_size, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(self.heads, 1, 1) + kv / scale.sqrt().view(self.heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output

    def forward(self, x, rel_pos, incremental_state=None):
        bsz, seq_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        g = self.g_proj(x)

        k *= self.scaling
        q = q.view(bsz, seq_len, self.heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.heads, self.key_dim).transpose(1, 2)

        qr = theta_shift(q, sin, cos)
        kr = theta_shift(k, sin, cos)

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)

        #output = self.group_norm(output.reshape(seq_len, self.head_size * self.heads)).reshape(bsz, seq_len, self.head_size * self.heads)
        output = self.group_norm(output).reshape(bsz, seq_len, self.head_size * self.heads)

        output = self.swish(g) * output

        output = self.out_proj(output)

        return output

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output