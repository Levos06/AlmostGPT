import numpy as np
import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pos_encoding = torch.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (10000 ** (2 * i / d_model)))
            for i in range(1, d_model, 2):
                pos_encoding[pos, i] = np.cos(pos / (10000 ** (2 * i / d_model)))
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, input_embedding):
        seq_len = input_embedding.size(1)
        # берем первые seq_len позиций и расширяем для батча
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0)
        # переносим на нужное устройство и приводим тип
        pos_enc = pos_enc.to(input_embedding.device).type(input_embedding.dtype)
        return input_embedding + pos_enc


