import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.model_v2 import QuerySenseBrain
    from src.tokenizer import CustomTokenizer
except ImportError:
    from model_v2 import QuerySenseBrain
    from tokenizer import CustomTokenizer

class QueryDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=64):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = self.tokenizer.encode(item['instruction'])
        trg = self.tokenizer.encode(item['response'])
        
        src = src[:self.max_len] + [0] * (self.max_len - len(src))
        trg = trg[:self.max_len] + [0] * (self.max_len - len(trg))
        
        return torch.tensor(src), torch.tensor(trg)

def train_model():
    torch.set_num_threads(4)
    device = torch.device("cpu")
    
    tokenizer = CustomTokenizer()
    tokenizer.load("models/vocab.json")
    
    data_path = "data/processed/clean_training.json"
    dataset = QueryDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = QuerySenseBrain(vocab_size=tokenizer.vocab_size).to(device)
    
    #  resume training logic 
    model_path = "models/v2_brain.pth"
    if os.path.exists(model_path):
        print(f"\n[INFO] Loading existing model from {model_path} to resume training...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("\n[INFO] No existing model found. Starting training from scratch...")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"--- Starting Training on {device} ---")
    print(f"Total Samples: {len(dataset)} | Total Batches: {len(dataloader)}")

    model.train()
    total_batches = len(dataloader)
    
    for epoch in range(1):
        total_loss = 0
        for batch_idx, (src, trg) in enumerate(dataloader):
            src, trg = src.to(device), trg.to(device)
            
            optimizer.zero_grad()
            
            output = model(src, trg[:, :-1])
            
            output = output.reshape(-1, output.shape[-1])
            trg_target = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg_target)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 50 == 0:
                percent = (batch_idx / total_batches) * 100
                print(f"Epoch [{epoch+1}] | Progress: {percent:.2f}% | Batch: {batch_idx}/{total_batches} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / total_batches
        print(f"==> Epoch {epoch+1} Complete! Average Loss: {avg_loss:.4f}")
      
    if not os.path.exists("models"): 
        os.makedirs("models")
    torch.save(model.state_dict(), "models/v2_brain.pth")
    print(f"\nSuccess! Model updated and saved as {model_path}")

if __name__ == "__main__":
    train_model()