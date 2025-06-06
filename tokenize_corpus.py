from datasets import load_dataset
from tokenizer import BPETokenizer

dataset = load_dataset("bentrevett/multi30k", split="train")  # загружаем один раз

for lang in ('en', 'de'):
    print(f"Training tokenizer for language: {lang}")
    text = ' '.join(example[lang] for example in dataset)

    tokenizer = BPETokenizer(max_vocab_size=5000)
    tokenizer.train(text)
    tokenizer.save_vocab(f"vocab_{lang}.json")
    print(f"Saved vocab_{lang}.json\n")




