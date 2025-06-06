import os
import json
import torch
from tokenizer import BPETokenizer
from my_dataset import TranslationDataset


def test_bpe_tokenizer():
    text = "hello hello hello world world"
    tokenizer = BPETokenizer(max_vocab_size=50)
    tokenizer.train(text)

    encoded = tokenizer.encode("hello world", add_special_tokens=True)
    decoded = tokenizer.decode([i for i in encoded if i not in (tokenizer.sos_id, tokenizer.eos_id)])

    print("Encoded with special tokens:", encoded)
    print("Decoded text:", decoded)

    assert tokenizer.pad_token in tokenizer.vocab
    assert tokenizer.sos_token in tokenizer.vocab
    assert tokenizer.eos_token in tokenizer.vocab
    assert tokenizer.unk_token in tokenizer.vocab
    assert decoded == "hello world", "Decode does not match original text"

    # Проверим сохранение и загрузку словаря
    tokenizer.save_vocab("test_vocab.json")
    new_tokenizer = BPETokenizer()
    new_tokenizer.load_vocab("test_vocab.json")
    assert new_tokenizer.vocab == tokenizer.vocab, "Vocab mismatch after loading"

    os.remove("test_vocab.json")
    print("BPETokenizer tests passed.\n")


def test_translation_dataset_and_collate():
    # Несколько предложений для теста
    src_texts = ["hello world", "goodbye world"]
    tgt_texts = ["hallo welt", "auf wiedersehen welt"]

    # Используем очень простой токенизатор, который знает только нужные символы
    tokenizer_src = BPETokenizer(max_vocab_size=20)
    tokenizer_src.train(' '.join(src_texts))
    tokenizer_tgt = BPETokenizer(max_vocab_size=20)
    tokenizer_tgt.train(' '.join(tgt_texts))

    dataset = TranslationDataset(src_texts, tgt_texts, tokenizer_src, tokenizer_tgt, max_len=10)

    # Получаем элементы
    for i in range(len(dataset)):
        src_ids, tgt_ids = dataset[i]
        print(f"Sample {i}:")
        print("  src_ids:", src_ids)
        print("  tgt_ids:", tgt_ids)
        assert src_ids.dtype == torch.long
        assert tgt_ids[0].item() == tokenizer_tgt.sos_id, "Target should start with <sos>"
        assert tgt_ids[-1].item() == tokenizer_tgt.eos_id, "Target should end with <eos>"

    # Теперь проверим collate_fn — он у тебя вынесен отдельно, я немного адаптирую его как функцию
    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)
        max_len_src = max(len(seq) for seq in src_batch)
        max_len_tgt = max(len(seq) for seq in tgt_batch)

        padded_src = torch.full((len(src_batch), max_len_src), tokenizer_src.pad_id, dtype=torch.long)
        padded_tgt = torch.full((len(tgt_batch), max_len_tgt), tokenizer_tgt.pad_id, dtype=torch.long)

        for i, (src_seq, tgt_seq) in enumerate(zip(src_batch, tgt_batch)):
            padded_src[i, :len(src_seq)] = src_seq
            padded_tgt[i, :len(tgt_seq)] = tgt_seq

        return padded_src, padded_tgt

    batch = [dataset[i] for i in range(len(dataset))]
    padded_src, padded_tgt = collate_fn(batch)
    print("\nPadded batch src shape:", padded_src.shape)
    print("Padded batch tgt shape:", padded_tgt.shape)

    # Проверим, что паддинг стоит на местах за пределами длины последовательностей
    for i in range(len(batch)):
        src_len = batch[i][0].shape[0]
        tgt_len = batch[i][1].shape[0]
        assert torch.all(padded_src[i, src_len:] == tokenizer_src.pad_id), "Src padding error"
        assert torch.all(padded_tgt[i, tgt_len:] == tokenizer_tgt.pad_id), "Tgt padding error"

    print("TranslationDataset and collate_fn tests passed.\n")


if __name__ == "__main__":
    test_bpe_tokenizer()
    test_translation_dataset_and_collate()
