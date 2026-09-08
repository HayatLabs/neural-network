import json
import os
from collections import Counter
import re

class CustomTokenizer:
    def __init__(self, vocab_size=30000):
        self.special_tokens = {
            "<PAD>": 0,  
            "<SOS>": 1,  
            "<EOS>": 2,  
            "<UNK>": 3  
        }
        self.vocab_size = vocab_size
        self.word2idx = self.special_tokens.copy()
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"([.,!?()])", r" \1 ", text) 
        return text

    def build_vocab(self, data_path):
        print(f"Tokenizer: Reading data from {data_path}...")
        word_counts = Counter()
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for item in data:
                full_text = self.clean_text(item['instruction'] + " " + item['response'])
                word_counts.update(full_text.split())

        most_common = word_counts.most_common(self.vocab_size - len(self.special_tokens))
        
        for word, _ in most_common:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        
        print(f"Tokenizer: Vocabulary built with {len(self.word2idx)} words.")

    def save(self, folder="models"):
        if not os.path.exists(folder): os.makedirs(folder)
        with open(os.path.join(folder, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(self.word2idx, f, ensure_ascii=False)
        print(f"Tokenizer: Vocab saved to {folder}/vocab.json")

    def load(self, path="models/vocab.json"):
        with open(path, 'r', encoding='utf-8') as f:
            self.word2idx = json.load(f)
            self.idx2word = {int(v): k for k, v in self.word2idx.items()}