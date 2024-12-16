from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from config import config
from huggingface_hub import HfApi
from typing import Optional
from datasets import Dataset
import shutil
import glob
import re
import json
import aiohttp
import logging
import asyncio

logger = logging.getLogger(__name__)

SUPPORTED_MODEL_TYPES = {     
    'sequence-generation': AutoModelForSeq2SeqLM,
    'text-generation': AutoModelForCausalLM
}

SUPPORTED_PIPELINES = {
    "text-generation",
    #"text2text-generation",
    #"summarization",
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
    "xglm": "text-generation"
}

def get_model_type(task: str):
    if task in ['translation']:
        return SUPPORTED_MODEL_TYPES['sequence-generation']
    elif task in ['text-generation']:
        return SUPPORTED_MODEL_TYPES['text-generation']
    else:
        raise ValueError(f"Unsupported task: {task}")



def setup_logging():
    base_dir = get_base_directory()
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)  

    log_file = os.path.join(log_dir, "hfts.log")

    # Configure a rotating file handler
    rotating_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=20, # Keep up to 20 backup files
        encoding="utf-8",
        delay=False
    )

    # Create a logging format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    rotating_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the desired logging level
    logger.addHandler(rotating_handler)
  
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Base Directory: {base_dir}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Log File: {log_file}")

def get_base_directory() -> str:
    """
    Returns the base directory where the executable is located.
    Handles both frozen (PyInstaller) and unfrozen states.
    """
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        return os.path.dirname(sys.executable)
    else:
        # If unfrozen, return the directory of the script file.
        return os.path.dirname(os.path.abspath(__file__))


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



def filter_unwanted_files(files):
    """Filter out unwanted files from the download list."""
    unwanted_files = {'.gitattributes', 'USE_POLICY.md'}
    
    # Check for unwanted filenames and exclude those containing '/' or '\'
    return [
        file for file in files 
        if file.rfilename not in unwanted_files and '/' not in file.rfilename and '\\' not in file.rfilename
    ]

def infer_model_type(model_path: str) -> str:
    """
    Infer model type based on the presence of *.gguf files, 'README.md', or 'config.json'.
   
    Priority:
    1. Parse 'README.md' for 'pipeline_tag' or 'tags'.
    2. Check for *.gguf files (indicates 'text-generation').
    3. Parse 'config.json' for 'model_type'.
   
    Parameters:
    - model_path (str): The full path to the model directory.
   
    Returns:
    - str: The inferred model type or 'unknown' if unable to determine.
    """
    logger.info(f"Inferring model type for path: {model_path}")
    
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
   
    logger.warning(f"Could not determine model type for {model_path}. Defaulting to 'unknown'.")
    return "unknown"

def get_directory_size(directory: str) -> int:
    """Calculate the total size of the directory."""
    logger.info(f"Calculating size for directory: {directory}")
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(directory):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    size = os.path.getsize(fp)
                    total_size += size
                    logger.info(f"Adding size of file '{fp}': {size} bytes")
    except Exception as e:
        logger.error(f"Error calculating directory size for {directory}: {e}")
    logger.info(f"Total size for directory '{directory}': {total_size} bytes")
    return total_size


def format_size(size_bytes: int) -> str:
    """
    Return a human-readable string representation of size in bytes,
    rounded to 0 decimal places.
    
    Parameters:
        size_bytes (int): The size in bytes.
        
    Returns:
        str: Formatted size string with appropriate units.
    """
    logger.debug(f"Formatting size: {size_bytes} bytes")
    if size_bytes == 0:
        logger.info(f"Formatted size: 0 Bytes")
        return "0 Bytes"
    
    size_units = ['Bytes', 'KB', 'MB', 'GB', 'TB']
    index = 0
    size = float(size_bytes)
    
    while size >= 1024 and index < len(size_units) - 1:
        size /= 1024
        index += 1
    
    formatted_size = f"{size:.0f} {size_units[index]}"
    logger.info(f"Formatted size: {formatted_size}")
    return formatted_size


def fetch_model_info(model_name: str, api_key: str):
    """Fetch model information from Hugging Face Hub."""
    logger.info(f"Fetching model info for '{model_name}' with API key provided.")
    api = HfApi()
    try:
        model_info = api.model_info(repo_id=model_name, token=api_key or os.getenv("HUGGINGFACE_TOKEN"))
        logger.debug(f"Fetched model info for '{model_name}': {model_info}")
        return model_info
    except Exception as e:
        logger.error(f"Failed to fetch model info for '{model_name}': {e}")
        raise
   
    
async def get_file_size_via_head(url: str, api_key: str) -> Optional[int]:
    """Retrieve the file size by performing a HEAD request with redirects allowed."""
    logger.info(f"Attempting HEAD request to {url} to get file size.")
    try:
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.head(url, allow_redirects=True) as response:
                if response.status == 200:
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        size = int(content_length)
                        logger.info(f"Content-Length for {url}: {size} bytes")
                        return size
                    else:
                        logger.warning(f"No Content-Length header for {url}")
                else:
                    logger.warning(f"Received status {response.status} for HEAD request to {url}")
    except Exception as e:
        logger.error(f"Failed to fetch size for URL {url}: {e}")
    return None

async def get_file_size_via_get(url: str, api_key: str) -> Optional[int]:
    """Retrieve the file size by performing a GET request with Range header."""
    logger.info(f"Attempting GET request to {url} with Range header to get file size.")
    try:
        headers = {'Range': 'bytes=0-0'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, allow_redirects=True) as response:
                if response.status in (200, 206):
                    content_range = response.headers.get('Content-Range')
                    if content_range:
                        size = content_range.split('/')[-1]
                        if size.isdigit():
                            size_int = int(size)
                            logger.info(f"Content-Range for {url}: {content_range} -> Size: {size_int} bytes")
                            return size_int
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        size = int(content_length)
                        logger.info(f"Content-Length for {url}: {size} bytes")
                        return size
                else:
                    logger.warning(f"Received status {response.status} for GET request to {url}")
    except Exception as e:
        logger.error(f"Failed to fetch size for URL {url} via GET: {e}")
    return None


async def run_map(dataset: Dataset, preprocess_function, client_id: str):
    """
    Apply a preprocessing function to a dataset.

    Parameters:
    - dataset (Dataset): The dataset to process.
    - preprocess_function (Callable): The preprocessing coroutine function.
    - client_id (str): Identifier for logging or tracking purposes.

    Returns:
    - Dataset: The processed dataset.
    """
    logger.info(f"Running map on dataset with client_id: {client_id}")
    from functools import partial

    async def async_preprocess(examples):
        logger.info(f"Preprocessing examples with client_id: {client_id}")
        try:
            result = await preprocess_function(examples)
            logger.info(f"Preprocessing successful for client_id: {client_id}")
            return result
        except Exception as e:
            logger.error(f"Error during preprocessing for client_id {client_id}: {e}")
            raise

    try:
        dataset = dataset.map(
            lambda examples: asyncio.run(async_preprocess(examples)),
            batched=True,
            remove_columns=['source', 'target']
        )
        logger.info(f"Map operation completed for client_id: {client_id}")
    except Exception as e:
        logger.error(f"Failed to run map on dataset for client_id {client_id}: {e}")
        raise
    return dataset