import numpy as np
from datasets import load_from_disk
from collections import Counter
import matplotlib.pyplot as plt

dataset = load_from_disk("ag_news_local")['train'][:]['text']
text = list(''.join(dataset))  # изначально каждый токен — 1 символ

ORIGINAL_LENGTH = len(''.join(dataset).split())
MAX_SIZE = 15000

# алфавит (набор токенов)
vocab = sorted(set(text))

num2char = {i: v for i, v in enumerate(vocab)}
char2num = {v: i for i, v in enumerate(vocab)}

num_vocab = np.array(list(range(0, len(vocab))))


def text2num(text):
    return np.array([char2num[i] for i in text])


num_text = text2num(text)


def merge_pair(tokens, pair_to_merge):
    new_tokens = []
    i = 0
    merged_token = num2char[pair_to_merge[0]] + num2char[pair_to_merge[1]]  # объединённый символ

    while i < len(tokens):
        if i < len(tokens) - 1 and (num2char[tokens[i]], num2char[tokens[i + 1]]) == (num2char[pair_to_merge[0]], num2char[pair_to_merge[1]]):
            new_tokens.append(char2num[merged_token])
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return np.array(new_tokens, dtype=int)


while len(num_text) / ORIGINAL_LENGTH > 2.5 and len(vocab) < MAX_SIZE:
    pairs_count = Counter(zip(num_text[:-1], num_text[1:]))

    most_common_pair = max(pairs_count.items(), key=lambda x: x[1])[0]
    print(f"Merging pair: {most_common_pair}, text length: {len(num_text)}, ratio: {len(num_text) / ORIGINAL_LENGTH:.4f}, vocab size: {len(vocab)}")

    new_token = num2char[most_common_pair[0]] + num2char[most_common_pair[1]]
    vocab.append(new_token)

    num2char = {i: v for i, v in enumerate(vocab)}
    char2num = {v: i for i, v in enumerate(vocab)}

    num_text = merge_pair(num_text, most_common_pair)


text = [num2char[i] for i in num_text]

print(text)

import json

# Сохраняем num2char (n2c)
with open('num2char.json', 'w', encoding='utf-8') as f:
    json.dump(num2char, f, ensure_ascii=False, indent=2)

# Сохраняем char2num (c2n)
with open('char2num.json', 'w', encoding='utf-8') as f:
    json.dump(char2num, f, ensure_ascii=False, indent=2)


