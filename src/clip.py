import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class ClipConfig:
    vocab_size = 49408
    d_model = 786
    seq_len = 77
    nb_layers = 12
    nb_heads = 12
    
    
class ClipEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_embeddings = nn.Parameter(torch.zeros((config.seq_len, config.d_model)))
    def forward(self, x):
        # x: (batch_size, seq_len)
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        x = self.input_embeddings(x)
        x += self.positional_embeddings
        return x
        
    
class ClipLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.attention = SelfAttention(config.n_heads, config.d_model)
        self.layernorm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.attention(self.layernorm1(x))
        x = x + self.mlp(self.layernorm2(x))
        return x
    
    
class CLIP(nn.Module):
    def __init__(self, config):
        self.embeddings = ClipEmbeddings(config)
        self.nb_layers = config.nb_layers
        
        self.encoder = nn.ModuleList(
            [
                ClipLayer(config) for i in range(self.nb_layers)
            ]
        )
        
        self.layernorm = nn.LayerNorm(config.d_model)
        
    def forward(self, input_ids: torch.LongTensor):
        # input_ids: (batch_size, seq_len)
        input_ids = input_ids.type(torch.Long)
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        embeddings = self.embeddings(input_ids)
        # batch_size, seq_len, d_model) -> batch_size, seq_len, d_model)
        for layer in self.encoder:
            embeddings = layer(embeddings)
        embeddings = self.layernorm(embeddings)
        
        return embeddings