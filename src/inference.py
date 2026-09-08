# src/inference.py
import torch
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.model_v2 import QuerySenseBrain
    from src.tokenizer import CustomTokenizer
except ImportError:
    from model_v2 import QuerySenseBrain
    from tokenizer import CustomTokenizer

class InferenceEngine:
    def __init__(self, model_path="models/v2_brain.pth", vocab_path="models/vocab.json"):
        self.device = torch.device("cpu")
        self.tokenizer = CustomTokenizer()
        self.tokenizer.load(vocab_path)
        self.model = QuerySenseBrain(vocab_size=self.tokenizer.vocab_size).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"--- QuerySense-AI Brain Loaded ---")
        else:
            print(f"Error: {model_path} not found!")
            sys.exit()
            
        self.model.eval()

    def generate_stream(self, instruction, max_len=100, temperature=0.8, top_k=50):
        src_tokens = self.tokenizer.encode(instruction)
        src = torch.tensor([src_tokens]).to(self.device)
        trg_tokens = [1] # <SOS>
        
        generated_text = ""
        
        for _ in range(max_len):
            trg = torch.tensor([trg_tokens]).to(self.device)
            with torch.no_grad():
                output = self.model(src, trg)
            
            logits = output[0, -1, :] / temperature
            
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float('Inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == 2: # <EOS>
                break
                
            trg_tokens.append(next_token)
            
            word = self.tokenizer.decode([next_token])
            yield word + " " 

def main():
    engine = InferenceEngine()
    
    print("\n" + "="*40)
    print("   🚀 QuerySense-AI V2: Neural Engine   ")
    print("="*40)
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("\nUser 👤: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input.strip():
            continue

        print("AI 🤖: ", end="", flush=True)
        
        for word in engine.generate_stream(user_input):
            print(word, end="", flush=True)
            time.sleep(0.05) 
        print("\n")

if __name__ == "__main__":
    main()