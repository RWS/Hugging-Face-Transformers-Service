import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import torch

# Ignore specific warnings
# Adjust the message to match the warning you want to suppress
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")
#warnings.filterwarnings("ignore", category=UserWarning)  # To ignore all UserWarnings

def load_model(model_path):
    # Load the tokenizer and model from the local path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    return tokenizer, model

def translate(text, tokenizer, model):
    # Tokenize the input text without specifying clean_up_tokenization_spaces
    inputs = tokenizer(text, return_tensors="pt")
    
    # Generate translation output
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    # Decode the generated tokens back to text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    # Specify the local path to the model
    model_path = "C:\\HuggingFace\\model_cache\\models--Helsinki-NLP--opus-mt-en-it\\snapshots\\be5a254d936f1a8c8da080406ca582a6615e6658"
    
    # Load the model and tokenizer
    tokenizer, model = load_model(model_path)
    
    # Example text to translate
    text_to_translate = "Hello, how are you?"

    # Perform translation
    translated = translate(text_to_translate, tokenizer, model)
    
    # Print the translated text
    print("Translated Text:", translated)