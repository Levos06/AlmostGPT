import torch
from torch import nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, num_heads)
        self.FF = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.MHA(x, x, x, mask)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        ff_output = self.dropout(self.FF(x))
        x = x + ff_output
        x = self.norm2(x)

        return x




