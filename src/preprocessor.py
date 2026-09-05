import re

class TextPreprocessor:
    def __init__(self):
        pass

    def clean(self, text):
       
        text = text.lower()
        
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = " ".join(text.split())
        print("print cleaned text => " + text)
        return text



    def tokenize(self, text):
        return text.split()