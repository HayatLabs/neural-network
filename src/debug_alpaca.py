import pandas as pd
import os

alpaca_path = 'data/alpaca_raw.parquet'

if os.path.exists(alpaca_path):
    print(f"--- Checking {alpaca_path} ---")
    try:
        df = pd.read_parquet(alpaca_path)
        
        print(f"Success! Total rows found: {len(df)}")
        print("\nColumns in this file:", df.columns.tolist())
        print("\nFirst 2 rows of data:")
        print(df[['instruction', 'output']].head(2))
        
        df.head(100).to_json('data/alpaca_sample.json', orient='records', indent=4)
        print("\nA sample of 100 rows saved to data/alpaca_sample.json")
        
    except Exception as e:
        print(f"Error reading Parquet: {e}")
else:
    print(f"File not found at {alpaca_path}. Please check the filename.")