import json
import os

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)