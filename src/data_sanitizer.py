import json
import os

def sanitize_data():
    input_file = 'data/processed/training_data.json' 
    output_file = 'data/processed/clean_training_v2.json'
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    print("Reading and cleaning 268k rows... This might take a minute.")
    clean_data = []

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        for item in data:
            instruction = item['instruction']
            response = item['response']
            
            if "##" in response:
                response = response.split("##")[0].strip()
            
            if "\n10. Instruction:" in response:
                response = response.split("\n10. Instruction:")[0].strip()
            
            if len(instruction) > 5 and len(response) > 5:
                clean_data.append({
                    "instruction": instruction.strip(),
                    "response": response.strip()
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)

    print(f"\nSanitization Complete!")
    print(f"Original Rows: {len(data)}")
    print(f"Cleaned Rows: {len(clean_data)}")
    print(f"File saved: {output_file}")

if __name__ == "__main__":
    sanitize_data()