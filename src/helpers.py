from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from config import config
import os
import shutil
import glob
import re
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = {     
    'sequence-generation': AutoModelForSeq2SeqLM,
    'text-generation': AutoModelForCausalLM
}

SUPPORTED_PIPELINES = {
    "text-generation",
    "text2text-generation",
    "summarization",
    "translation"
}

# Used to infer the model type from the model_type property in the config file 
# when readme is not present and/or not defined in the pipeline or tags
# This is not meant to be a definitive list
MODEL_TYPE_MAPPING = {
    "m2m_100": "translation",
    "marian": "translation",
    "mbart": "translation",
    "mistral": "text-generation",
    "qwen2": "text-generation",
    "t5": "translation",   
}

def get_model_type(task: str):
    if task in ['translation']:
        return SUPPORTED_MODEL_TYPES['sequence-generation']
    elif task in ['text-generation']:
        return SUPPORTED_MODEL_TYPES['text-generation']
    else:
        raise ValueError(f"Unsupported task: {task}")
    

def extract_assistant_response(response):
    """
    Extracts the content from the first 'assistant' role in the response.

    Supports different response structures to ensure compatibility with various LLMs.
    """
    # Handle list responses by taking the first element
    if isinstance(response, list) and len(response) > 0:
        response = response[0]

    if isinstance(response, dict):
        # Handling for llama-cpp-python and similar models
        if 'choices' in response:
            choices = response['choices']
            for choice in choices:
                message = choice.get('message', {})
                if message.get('role') == 'assistant':
                    content = message.get('content')
                    if content:
                        return content.strip()
        
        # Handling for other models that use 'generated_text'
        elif 'generated_text' in response:
            generated_text = response.get('generated_text')
            if isinstance(generated_text, list):
                for entry in generated_text:
                    if entry.get('role') == 'assistant':
                        content = entry.get('content')
                        if content:
                            return content.strip()
            elif isinstance(generated_text, str):
                return generated_text.strip()

    # Fallback: return the entire response as string if expected keys are missing
    return str(response)


def move_snapshot_files(model_name: str, download_directory: str):
    """Move snapshot files from the downloaded model's snapshots directory to its root."""
    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))
    snapshots_path = os.path.join(model_path, "snapshots")
    
    if os.path.exists(snapshots_path):
        logger.info(f"Processing snapshots for model: {model_name}")
        
        # Iterate over items in the model_path
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            
            # Skip the 'snapshots' directory
            if item == 'snapshots':
                continue
            
            if os.path.isdir(item_path):
                # If it's a directory, remove it and its contents
                try:
                    shutil.rmtree(item_path)
                    logger.info(f"Removed directory: {item_path}")
                except Exception as e:
                    logger.error(f"Failed to remove directory {item_path}: {e}")
            elif os.path.isfile(item_path):
                # If it's a file, remove it
                try:
                    os.remove(item_path)
                    logger.info(f"Removed file: {item_path}")
                except Exception as e:
                    logger.error(f"Failed to remove file {item_path}: {e}")
            else:
                logger.warning(f"Skipped unknown item type: {item_path}")
        
        # Handle snapshot directories
        snapshot_dirs = glob.glob(os.path.join(snapshots_path, '*'))
        if snapshot_dirs:
            first_snapshot = snapshot_dirs[0] 
            logger.info(f"Moving files from snapshot: {first_snapshot}")
            
            for item in os.listdir(first_snapshot):
                source_path = os.path.join(first_snapshot, item)
                destination_path = os.path.join(model_path, item)
                
                # Skip if the destination already exists
                if os.path.exists(destination_path):
                    logger.info(f"Destination already exists, skipping: {destination_path}")
                    continue
                
                try:
                    shutil.move(source_path, destination_path)
                    logger.info(f"Moved {source_path} to {destination_path}")
                except Exception as e:
                    logger.error(f"Failed to move {source_path} to {destination_path}: {e}")
            
            # Remove the snapshots directory after moving files
            try:
                shutil.rmtree(snapshots_path)
                logger.info(f"Removed snapshots directory: {snapshots_path}")
            except Exception as e:
                logger.error(f"Failed to remove snapshots directory {snapshots_path}: {e}")
    else:
        logger.warning(f"No snapshots directory found for model: {model_name}")



def filter_unwanted_files(files):
    """Filter out unwanted files from the download list."""
    unwanted_files = {'.gitattributes', 'USE_POLICY.md'}
    
    # Check for unwanted filenames and exclude those containing '/' or '\'
    return [
        file for file in files 
        if file.rfilename not in unwanted_files and '/' not in file.rfilename and '\\' not in file.rfilename
    ]


def infer_model_type(model_dir: str, download_directory: str) -> str:
    """
    Infer model type based on the presence of *.gguf files, 'README.md', or 'config.json'.
    
    Priority: 
    1. Parse 'README.md' for 'pipeline_tag' or 'tags'.
    2. Check for *.gguf files (indicates 'text-generation').
    3. Parse 'config.json' for 'model_type'.
    
    Parameters:
    - model_dir (str): The directory name of the model.
    - download_directory (str): The base path where models are stored.
    
    Returns:
    - str: The inferred model type or 'unknown' if unable to determine.
    """
    model_path = os.path.join(download_directory, model_dir)

    # Parse README.md
    readme_path = os.path.join(model_path, 'README.md')
    if os.path.exists(readme_path):
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read()
                
                # Extract pipeline_tag
                pipeline_tag_match = re.search(r'pipeline_tag:\s*(\S+)', readme_content, re.IGNORECASE)
                if pipeline_tag_match:
                    pipeline_tag = pipeline_tag_match.group(1).strip().lower()
                    if pipeline_tag in SUPPORTED_PIPELINES:
                        logger.info(f"Found pipeline_tag '{pipeline_tag}' in README.md. Model type: '{pipeline_tag}'")
                        return pipeline_tag
                
                # Extract tags
                tags_match = re.search(r'tags:\s*\n((?:\s*-\s*\w+)+)', readme_content, re.IGNORECASE)
                if tags_match:
                    tags_block = tags_match.group(1)
                    tags = re.findall(r'-\s*(\w+)', tags_block)
                    for tag in tags:
                        tag_lower = tag.lower()
                        if tag_lower in SUPPORTED_PIPELINES:
                            logger.info(f"Found tag '{tag_lower}' in README.md. Model type: '{tag_lower}'")
                            return tag_lower
        except Exception as e:
            logger.error(f"Error reading README.md in {model_path}: {e}")
    
    # Check for .gguf files
    gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
    if gguf_files:
        logger.info(f"Detected .gguf files in {model_path}. Model type: 'text-generation'")
        return "text-generation"
    
    # Parse config.json
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_file = json.load(f)
                model_type = config_file.get("model_type", "").lower().strip()
                
                # Check within 'text_config' if necessary
                if not model_type and "text_config" in config_file:
                    model_type = config_file["text_config"].get("model_type", "").lower().strip()
                
                inferred_type = MODEL_TYPE_MAPPING.get(model_type)
                if inferred_type:
                    logger.info(f"Derived model_type '{model_type}' from config.json. Model type: '{inferred_type}'")
                    return inferred_type
        except Exception as e:
            logger.error(f"Error reading config.json in {model_path}: {e}")
    
    logger.warning(f"Could not determine model type for {model_dir}. Defaulting to 'unknown'.")
    return "unknown"


def get_directory_size(directory: str) -> int:
    """Calculate the total size of the directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):  # Check if the file exists
                total_size += os.path.getsize(fp)
    return total_size

def format_size(size_bytes: int) -> str:
    """Return a human-readable string representation of size in bytes."""
    if size_bytes == 0:
        return "0 Bytes"
    size_units = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    index = 0
    while size_bytes >= 1024 and index < len(size_units) - 1:
        size_bytes /= 1024
        index += 1
    return f"{size_bytes:.2f} {size_units[index]}"



