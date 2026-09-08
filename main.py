# main.py
from src.utils import load_config, ensure_dir
from src.preprocessor import TextPreprocessor
from src.classifier import IntentClassifier
import pandas as pd

def main():
    config = load_config()
    ensure_dir(config['paths']['model_dir'])
    
    preprocessor = TextPreprocessor()
    classifier = IntentClassifier(model_dir=config['paths']['model_dir'])

    training_data = [
        ("gaming laptop price", "product_search"),
        ("50k er moddhe bhalo phone", "product_search"),
        ("how to fix laptop hanging", "problem_solving"),
        ("windows update error 0x800", "problem_solving"),
        ("iphone 15 vs samsung s24", "comparison"),
        ("intel or amd which is best", "comparison"),
        ("suggest a good book to read", "recommendation"),
        ("best budget headphones 2024", "recommendation"),
        ("who is the ceo of tesla", "information_question"),
        ("what is the weather in dhaka", "information_question")
    ]
    
    df = pd.DataFrame(training_data, columns=['text', 'intent'])

    print(f"--- Starting {config['project_name']} Training ---")
    df['text'] = df['text'].apply(preprocessor.clean)
    classifier.train(df['text'], df['intent'])
    print("Training complete!\n")

    print("Enter your query to test the engine (type 'exit' to quit):")
    while True:
        query = input(">> ")
        if query.lower() == 'exit':
            break
        
        clean_query = preprocessor.clean(query)
        result = classifier.predict(clean_query)
        
        print(f"Result: {result['intent']} (Confidence: {result['confidence']})")

if __name__ == "__main__":
    main()