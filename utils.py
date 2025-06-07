import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_pad_mask(seq, pad_token_id):
    """
    Возвращает маску паддингов: True — это паддинг, False — это токен.
    """
    return (seq == pad_token_id).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]


def generate_causal_mask(seq_len):
    """
    Возвращает нижнетреугольную маску для причинности (decoder self-attention).
    """
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))  # [seq_len, seq_len]


def combine_masks(pad_mask, causal_mask):
    """
    Объединяет паддинг-маску и причинную маску.
    """
    return pad_mask.to(device) & causal_mask.to(device).unsqueeze(0).unsqueeze(1)  # broadcasting to [batch, 1, seq_len, seq_len]


def generate_src_mask(src, pad_token_id):
    return generate_pad_mask(src, pad_token_id)


def generate_tgt_mask(tgt, pad_token_id):
    pad_mask = generate_pad_mask(tgt, pad_token_id)         # [batch, 1, 1, seq_len]
    causal_mask = generate_causal_mask(tgt.size(1))         # [seq_len, seq_len]
    return combine_masks(pad_mask, causal_mask)             # [batch, 1, seq_len, seq_len]

