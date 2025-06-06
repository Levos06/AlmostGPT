from torch import nn
from transformer_decoder_layer import TransformerDecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output, tgt_mask, memory_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)

        return self.final_norm(x)









