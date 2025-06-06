from torch import nn
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.MHCA = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.FF = FeedForward(d_model, d_ff)

    def forward(self, x, encoder_output, self_attn_mask, cross_attn_mask):
        attention_output = self.MHA(x, x, x, self_attn_mask)
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        cross_attention_output = self.MHCA(x, encoder_output, encoder_output, cross_attn_mask)
        x = x + self.dropout(cross_attention_output)
        x = self.norm2(x)

        ff_output = self.FF(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x











