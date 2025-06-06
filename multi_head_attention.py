import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.size()

        # Линейные проекции
        Q = self.q_proj(query)  # [batch, seq_len, heads, head_dim]
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Разбиение на головы
        def split_heads(x):
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Q = split_heads(Q)  # [batch, heads, seq_len, head_dim]
        K = split_heads(K)
        V = split_heads(V)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, heads, seq_len, seq_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)  # [batch, heads, seq_len, head_dim]

        # Объединение голов
        concat = attention_output.permute(0, 2, 1, 3).contiguous()  # [batch, seq_len, heads, head_dim]
        concat = concat.view(batch_size, seq_len, self.d_model)     # [batch, seq_len, d_model]

        # Финальная проекция
        output = self.out_proj(concat)

        return output
