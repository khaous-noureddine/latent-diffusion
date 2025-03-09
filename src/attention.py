import torch
import torch.nn as nn 
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_model, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias=in_proj_bias)
        self.Wk = nn.Linear(d_model, d_model, bias=in_proj_bias)
        self.Wv = nn.Linear(d_model, d_model, bias=in_proj_bias)
        self.Wo = nn.Linear(d_model, d_model, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_model = d_model
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.dv = d_model // n_heads
        
    def forward(self, x, causal_mask=True):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        # Three times : (batch_size, seq_len, d_model -> batch_size, seq_len, d_model)
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        
        multiheads_shape = (batch_size, seq_len, self.n_heads, self.dv)
        
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, dv) -> (batch_size, n_heads, seq_len, dv)
        q = q.view(multiheads_shape).transpose(1, 2) 
        k = k.view(multiheads_shape).transpose(1, 2)
        v = v.view(multiheads_shape).transpose(1, 2)
        
        # (batch_size, n_heads, seq_len, dv) @ (batch_size, n_heads, dv, seq_len) -> (batch_size, n_heads, seq_len, seq_len)
        attention = q @ k.transpose(-1, -2)
        attention /= torch.sqrt(self.dv)
        attention = F.softmax(attention, dim=-1)
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, dv) -> (batch_size, n_heads, seq_len, dv)
        attention = attention @ v
        
        # (batch_size, n_heads, seq_len, dv) -> (batch_size, seq_len, n_heads, dv)
        attention = attention.transpose(1, 2)
        # (batch_size, seq_len, n_heads, dv) -> (batch_size, seq_len, d_model)
        attention = attention.reshape((batch_size, seq_len, d_model))
        # batch_size, seq_len, d_model -> batch_size, seq_len, d_model)
        attention = self.Wo(attention)
        
        return attention
        
        
        