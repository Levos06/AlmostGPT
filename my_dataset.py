import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, tokenizer_src, tokenizer_tgt, max_len=64):
        assert len(src_lines) == len(tgt_lines)
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_len = max_len

        self.pad_id_src = tokenizer_src.pad_id
        self.pad_id_tgt = tokenizer_tgt.pad_id
        self.sos_id_tgt = tokenizer_tgt.sos_id
        self.eos_id_tgt = tokenizer_tgt.eos_id

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        # Токенизируем вход (английский)
        src_ids = self.tokenizer_src.encode(self.src_lines[idx], add_special_tokens=False)
        # Токенизируем выход (немецкий) и добавляем <sos> и <eos>
        tgt_ids = self.tokenizer_tgt.encode(self.tgt_lines[idx], add_special_tokens=False)
        tgt_ids = [self.sos_id_tgt] + tgt_ids + [self.eos_id_tgt]

        # Усечение до max_len
        if len(src_ids) > self.max_len:
            src_ids = src_ids[:self.max_len]
        if len(tgt_ids) > self.max_len:
            tgt_ids = tgt_ids[:self.max_len]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def collate_fn(self, batch):
    src_batch, tgt_batch = zip(*batch)

    # Определяем максимальную длину в батче
    max_len_src = max(len(seq) for seq in src_batch)
    max_len_tgt = max(len(seq) for seq in tgt_batch)

    # Создаем паддинговые тензоры
    padded_src = torch.full((len(src_batch), max_len_src), self.pad_id_src, dtype=torch.long)
    padded_tgt = torch.full((len(tgt_batch), max_len_tgt), self.pad_id_tgt, dtype=torch.long)

    # Копируем значения
    for i, (src_seq, tgt_seq) in enumerate(zip(src_batch, tgt_batch)):
        padded_src[i, :len(src_seq)] = src_seq
        padded_tgt[i, :len(tgt_seq)] = tgt_seq

    return padded_src, padded_tgt

