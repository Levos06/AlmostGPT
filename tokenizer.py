import numpy as np
from collections import Counter
import json
from tqdm import tqdm

class BPETokenizer:
    def __init__(self, max_vocab_size=20000):
        self.max_vocab_size = max_vocab_size
        self.vocab = []
        self.char2num = {}
        self.num2char = {}
        self.num_text = None
        self.original_length = None

    def _init_vocab(self, text):
        text = list(text)
        self.vocab = sorted(set(text))
        self.char2num = {v: i for i, v in enumerate(self.vocab)}
        self.num2char = {i: v for i, v in enumerate(self.vocab)}
        self.num_text = np.array([self.char2num[c] for c in text])
        self.original_length = len(''.join(text).split())

    def _merge_pair(self, tokens, pair):
        new_tokens = []
        i = 0
        merged_token = self.num2char[pair[0]] + self.num2char[pair[1]]

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                new_tokens.append(self.char2num[merged_token])
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return np.array(new_tokens)

    def train(self, text):
        self._init_vocab(text)
        pbar = tqdm(desc="Training BPE", total=self.max_vocab_size - len(self.vocab))

        while len(self.vocab) < self.max_vocab_size:
            pairs_count = Counter(zip(self.num_text[:-1], self.num_text[1:]))

            if not pairs_count:
                break

            most_common_pair = max(pairs_count.items(), key=lambda x: x[1])[0]
            merged_token = self.num2char[most_common_pair[0]] + self.num2char[most_common_pair[1]]

            # Добавляем в словарь
            self.vocab.append(merged_token)
            self.char2num[merged_token] = len(self.vocab) - 1
            self.num2char[len(self.vocab) - 1] = merged_token

            # Объединяем токены
            self.num_text = self._merge_pair(self.num_text, most_common_pair)

            pbar.set_postfix({
                "ratio": f"{len(self.num_text)/self.original_length:.2f}",
                "vocab": len(self.vocab)
            })
            pbar.update(1)

        pbar.close()

    def encode(self, text):
        tokens = [self.char2num[c] for c in text]
        return tokens

    def decode(self, tokens):
        return ''.join(self.num2char[i] for i in tokens)

    def save_vocab(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab": self.vocab,
                "char2num": self.char2num,
                "num2char": {str(k): v for k, v in self.num2char.items()}
            }, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.char2num = data["char2num"]
        self.num2char = {int(k): v for k, v in data["num2char"].items()}


