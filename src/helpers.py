from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from llama_cpp import Llama
from config import config
import os
import shutil
import glob
import re
import json


# Supported model mappings
SUPPORTED_MODEL_TYPES = {     
    'sequence-generation': AutoModelForSeq2SeqLM,
    'text-generation': AutoModelForCausalLM,
    'llama': Llama
}

# Function to get model type based on task
def get_model_type(task: str):
    if task in ['translation', 'text2text-generation', 'summarization']:
        return SUPPORTED_MODEL_TYPES['sequence-generation']
    elif task in ['text-generation']:
        return SUPPORTED_MODEL_TYPES['text-generation']
    elif task in ['llama']:
        return SUPPORTED_MODEL_TYPES['llama']
    else:
        raise ValueError(f"Unsupported task: {task}")
    


def extract_assistant_response(response):
    """
    Function to extract the most recent assistant response from a structured output.
    Assumes the structure contains a list where each entry may have a 'role' and 'content'.
    """
    # Assuming the response has a 'generated_text' key referring to a list of responses
    if isinstance(response, dict) and 'generated_text' in response:
        generated_text_list = response['generated_text']
        
        if isinstance(generated_text_list, list):
            # Filter entries with role 'assistant'
            assistant_responses = [entry for entry in generated_text_list if entry.get('role') == 'assistant']
            if assistant_responses:
                # Return the content of the last assistant's response
                return assistant_responses[-1].get('content', '')
    
    return None


# Function to move snapshot files to the model path
def move_snapshot_files(model_name, download_directory):
    """Move snapshot files from the downloaded model's snapshots directory to its root."""
    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))
    snapshots_path = os.path.join(model_path, "snapshots")

    if os.path.exists(snapshots_path):
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if item != 'snapshots' and os.path.exists(item_path):
                shutil.rmtree(item_path)  

        snapshot_dirs = glob.glob(os.path.join(snapshots_path, '*'))
        if snapshot_dirs:
            first_snapshot = snapshot_dirs[0]  
            for item in os.listdir(first_snapshot):
                source_path = os.path.join(first_snapshot, item)
                destination_path = os.path.join(model_path, item)

                if os.path.exists(destination_path):
                    continue  
                
                shutil.move(source_path, model_path)
            shutil.rmtree(snapshots_path)

def filter_unwanted_files(files):
    """Filter out unwanted files from the download list."""
    unwanted_files = {'.gitattributes', 'USE_POLICY.md'}
    
    # Check for unwanted filenames and exclude those containing '/' or '\'
    return [
        file for file in files 
        if file.rfilename not in unwanted_files and '/' not in file.rfilename and '\\' not in file.rfilename
    ]

def infer_model_type(model_dir: str) -> str:
    """Infer model type based on the 'config.json' or 'README.md' contents."""
    
    model_cache_dir = config.DOWNLOAD_DIRECTORY
    model_path = os.path.join(model_cache_dir, model_dir)

    # Construct path for README.md
    readme_path = os.path.join(model_path, 'README.md')

    # Check the README for specific information
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
            
            # Use regex to find the pipeline_tag value
            pipeline_tag_match = re.search(r'pipeline_tag:\s*(\S+)', readme_content)
            if pipeline_tag_match:
                pipeline_tag_value = pipeline_tag_match.group(1).strip()                         
                if pipeline_tag_value in ("text-generation", "text2text-generation", "summarization", "translation"):
                    return pipeline_tag_value
                
            # Search for tags specifically
            tags_match = re.search(r'tags:\s*-\s*([\w-]+(?:\n- [\w-]+)*)', readme_content)
            if tags_match:
                tags_list = tags_match.group(0).splitlines()
                # Look for specific entries in the tags list
                for tag in tags_list:
                    tag_value = tag.strip().replace('- ', '').strip()  # Clean tag value
                    if tag_value in ("text-generation", "text2text-generation", "summarization", "translation"):
                         return tag_value
                            

    # Then, check in the config.json
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config_file = json.load(f)
            model_type = config_file.get("model_type", "").lower().strip()

            # If not found at the top level, check within text_config if it exists
            if not model_type and "text_config" in config_file:
                model_type = config_file["text_config"].get("model_type", "").lower().strip()

            # Infer from model_type
            if model_type in ["m2m_100", "marian", "mbart", "mistral","qwen2", "t5"]:
                return "translation"     
            elif model_type in ["phi3", "mistral"]:
                return "text-generation"        

    # Final check: Look for llama compatible files with the *.gguf extension
    gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
    if gguf_files:
        return "llama"
    
    return "unknown"  # Fallback if type cannot be determined

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



