from torch import nn
from positional_encoding import PositionalEncoding
from transformer_decoder import TransformerDecoder
from transformer_encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, d_model, src_vocab_size, tgt_vocab_size, max_len, d_ff, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = TransformerEncoder(d_model, d_ff, num_heads, num_layers)
        self.decoder = TransformerDecoder(d_model, d_ff, num_heads, num_layers)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask, tgt_mask, memory_mask):
        # Эмбеддинги + позиционное кодирование + дропаут
        src_vec = self.dropout(self.pos_encoding(self.src_embedding(src)))
        tgt_vec = self.dropout(self.pos_encoding(self.tgt_embedding(tgt)))

        # Прогон через энкодер
        memory = self.encoder(src_vec, src_mask)

        # Прогон через декодер
        decoder_output = self.decoder(tgt_vec, memory, tgt_mask, memory_mask)

        # Финальный линейный слой — логиты
        logits = self.output_linear(decoder_output)

        return logits















