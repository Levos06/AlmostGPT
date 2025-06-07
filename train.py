import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from collate import collate_fn
from tokenizer import BPETokenizer
from transformer import Transformer
from utils import generate_src_mask, generate_tgt_mask
from my_dataset import TranslationDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Гиперпараметры
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1
max_len = 128
src_vocab_size = 219
tgt_vocab_size = 343
batch_size = 32
num_epochs = 10
lr = 1e-4

# Модель
model = Transformer(
    d_model=d_model,
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    max_len=max_len,
    d_ff=d_ff,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout
).to(device)

# Оптимизатор и функция потерь
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=0)

dataset = load_dataset("bentrevett/multi30k", split="train")
src_lines = [example['en'] for example in dataset][:10]
tgt_lines = [example['de'] for example in dataset][:10]
tokenizer_src = BPETokenizer()
tokenizer_src.load_vocab("vocab_en.json")
tokenizer_tgt = BPETokenizer()
tokenizer_tgt.load_vocab("vocab_de.json")

# Загрузка данных
train_dataset = TranslationDataset(src_lines, tgt_lines, tokenizer_src, tokenizer_tgt, max_len=128)
collate = collate_fn(
    pad_id_src=tokenizer_src.pad_id,
    pad_id_tgt=tokenizer_tgt.pad_id
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    i = 0
    for src, tgt in train_loader:
        i += 1
        print(f"{i}/{len(train_loader)}")
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = generate_src_mask(src, 0).to(device)
        tgt_mask = generate_tgt_mask(tgt_input, 0).to(device)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_mask)

        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

PATH = '/content/drive/MyDrive/my_transformer.pth'
torch.save(model.state_dict(), PATH)
print(f"Модель сохранена по пути: {PATH}")
