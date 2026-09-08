# src/data_processor.py
import pandas as pd
import json
import os

def process_data():
    processed_data = []

    dolly_path = 'data/dolly_raw.jsonl'
    alpaca_path = 'data/alpaca_raw.parquet'
    output_path = 'data/processed/training_data.json'

    if os.path.exists(dolly_path):
        print(f"Processing Dolly dataset from {dolly_path}...")
        with open(dolly_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                instruction = data['instruction']
                if data.get('context'):
                    instruction += " [Context]: " + data['context']
                
                processed_data.append({
                    "instruction": instruction,
                    "response": data['response']
                })
    else:
        print(f"Error: {dolly_path} not found!")

    if os.path.exists(alpaca_path):
        print(f"Processing Alpaca dataset from {alpaca_path}...")
        df = pd.read_parquet(alpaca_path)
        for _, row in df.iterrows():
            instruction = row['instruction']
            if row.get('input'):
                instruction += " [Input]: " + row['input']
            
            processed_data.append({
                "instruction": instruction,
                "response": row['output']
            })
    else:
        print(f"Error: {alpaca_path} not found!")

    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
        
    if processed_data:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        print(f"\nSuccess! Total {len(processed_data)} samples processed.")
        print(f"File saved at: {output_path}")
    else:
        print("No data processed. Check if the raw files are in the 'data' folder.")

if __name__ == "__main__":
    process_data()