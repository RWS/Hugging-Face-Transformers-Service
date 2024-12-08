import os
import pandas as pd
from datasets import Dataset
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration, Trainer, TrainingArguments
import torch

# Define paths
model_path = "C:/HuggingFace/Models/facebook--mbart-large-50-many-to-many-mmt"
output_dir = "C:/HuggingFace/Models/facebook--mbart-large-50-many-to-many-mmt-finetuned-it"
data_file = "C:/HuggingFace/Data/data.csv"

# Validate paths
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

if not os.path.exists(data_file):
    raise FileNotFoundError(f"Data file '{data_file}' does not exist.")

# Load the tokenizer and model using specific classes
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path)
print("Tokenizer and model loaded successfully.")

# Define language codes
source_lang = "en_XX"
target_lang = "it_IT"

# Validate language codes
if source_lang not in tokenizer.lang_code_to_id:
    raise ValueError(f"Unsupported source language code: '{source_lang}'")

if target_lang not in tokenizer.lang_code_to_id:
    raise ValueError(f"Unsupported target language code: '{target_lang}'")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory '{output_dir}' is ready.")

# Load and prepare the dataset
df = pd.read_csv(data_file)
print("Data file loaded into DataFrame.")

# Ensure the CSV has 'src_text' and 'tgt_text' columns
if 'src_text' not in df.columns or 'tgt_text' not in df.columns:
    raise ValueError("CSV data file must contain 'src_text' and 'tgt_text' columns.")

# Clean and validate 'src_text' and 'tgt_text'
df['src_text'] = df['src_text'].astype(str).fillna('').str.strip()
df['tgt_text'] = df['tgt_text'].astype(str).fillna('').str.strip()
print("Converted 'src_text' and 'tgt_text' to strings and handled missing values.")

# Remove rows where 'tgt_text' is empty
initial_count = len(df)
df = df[df['tgt_text'] != '']
final_count = len(df)
print(f"Filtered dataset: {final_count} out of {initial_count} examples remain.")

if final_count == 0:
    raise ValueError("No valid examples found in the dataset after filtering out empty 'tgt_text'.")

# Prepare the dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.rename_column("src_text", "source")
dataset = dataset.rename_column("tgt_text", "target")
print("Dataset prepared and columns renamed.")

def preprocess_function(examples):
    inputs = examples["source"]
    targets = examples["target"]

    # Prepend target language token to the inputs to specify translation direction
    inputs = [f">>{target_lang}<< {text}" for text in inputs]

    # Ensure all targets are non-empty strings
    targets = [text if isinstance(text, str) and text.strip() else " " for text in targets]

    # Log samples for debugging
    if len(inputs) > 0:
        print(f"Preprocessed Input Sample: {inputs[0]}")
        print(f"Input Sample Type: {type(inputs[0])}")
    if len(targets) > 0:
        print(f"Preprocessed Target Sample: {targets[0]}")
        print(f"Target Sample Type: {type(targets[0])}")

    # Set source and target languages
    tokenizer.src_lang = source_lang
    tokenizer.tgt_lang = target_lang  # Explicitly set tgt_lang
    print(f"Set tokenizer src_lang='{source_lang}' and tgt_lang='{target_lang}'.")

    # Tokenize the inputs
    try:
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding="max_length"
        )
        print("Input texts tokenized.")
    except Exception as e:
        print(f"Error tokenizing inputs: {e}")
        raise e

    # Tokenize the targets
    try:
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding="max_length",
            #text_target=True  # Replaces 'as_target_tokenizer'
        )
        print("Target texts tokenized.")
    except Exception as e:
        print(f"Error tokenizing targets: {e}")
        raise e

    # Assign labels
    if "input_ids" not in labels:
        print("Labels do not contain 'input_ids'.")
        raise ValueError("Tokenization of targets failed.")

    # Check for None in labels
    for i, label in enumerate(labels["input_ids"]):
        if label is None:
            print(f"Label at index {i} is None.")
            raise ValueError(f"Label at index {i} is None.")

    model_inputs["labels"] = labels["input_ids"]
    print("Labels assigned to model inputs.")
    return model_inputs

# Apply the preprocessing
print("Applying preprocessing to the dataset.")
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['source', 'target'])
tokenized_dataset.set_format("torch")
print("Dataset tokenization complete.")

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=4, 
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=2,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    save_strategy="steps",
    evaluation_strategy="no",  
    predict_with_generate=False, 
    fp16=torch.cuda.is_available(),
)
print("Training arguments set.")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # eval_dataset=validation_dataset,  # Define if you have a validation set
)
print("Trainer initialized.")

print("Commencing training.")
trainer.train()
print("Training completed.")

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Fine-tuned model and tokenizer saved to '{output_dir}'.")