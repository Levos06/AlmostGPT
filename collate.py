import torch


def collate_fn(pad_id_src, pad_id_tgt):
    def _collate(batch):
        src_batch, tgt_batch = zip(*batch)

        max_len_src = max(len(seq) for seq in src_batch)
        max_len_tgt = max(len(seq) for seq in tgt_batch)

        padded_src = torch.full((len(batch), max_len_src), pad_id_src, dtype=torch.long)
        padded_tgt = torch.full((len(batch), max_len_tgt), pad_id_tgt, dtype=torch.long)

        for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
            padded_src[i, :len(src)] = src
            padded_tgt[i, :len(tgt)] = tgt

        return padded_src, padded_tgt

    return _collate
