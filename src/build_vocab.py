from src.tokenizer import CustomTokenizer

import os

def main():
    data_path = 'data/processed/clean_training.json'
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return

    tokenizer = CustomTokenizer(vocab_size=30000) 
    tokenizer.build_vocab(data_path)
    tokenizer.save()

if __name__ == "__main__":
    main()