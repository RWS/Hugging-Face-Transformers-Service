from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from huggingface_hub import hf_hub_download, model_info
import torch
import os
import glob
import json
import asyncio
from dotenv import load_dotenv
import warnings
import shutil
from typing import Optional, List, Dict 
import re


# Load environment variables
load_dotenv()
# Suppress specific Pydantic warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

app = FastAPI(
    title="Hugging Face Transformers Service",
    description="This is a FastAPI application designed to provide an intuitive and efficient interface for working with Hugging Face models, specifically catering to translation and text generation tasks. The service allows users to **download and mount** models locally, making it possible to run model inference without requiring an internet connection once the models are downloaded.",
    version="1.0.0",
)

# Supported model mappings
SUPPORTED_MODEL_TYPES = {
    'translation': AutoModelForSeq2SeqLM,
    'text2text-generation': AutoModelForSeq2SeqLM,
    'text-generation': AutoModelForCausalLM,
}


class TranslationRequest(BaseModel):
    text: str = Field(default="The cat is on the table.", description="The source content that should be translated")
    
    class Config:
        protected_namespaces = ()  
        json_schema_extra  = {
            "example": {
                "text": "The cat is on the table."
            }
        }

class TextGenerationRequest(BaseModel):
    prompt: List[Dict[str, str]] = Field(
        default=[
            {"role": "system", "content": "you are a helpful translator"},
            {"role": "user", "content": "translate this from English to Italian: The cat is on the table."}
        ],
        description="The User Prompt comprises of custom instructions provided by the user, detailing the specific requirements for translation."
    )
    max_tokens: int = Field(default=250, description="The maximum number of tokens to generate.")
    temperature: float = Field(default=1.0, description="Sampling temperature for generation, where higher values lead to more random outputs.")
    
    class Config:
        protected_namespaces = ()  
        json_schema_extra = {
            "example": {
                "prompt": [
                    {"role": "system", "content": "you are a helpful translator"},
                    {"role": "user", "content": "translate this from English to Italian: The cat is on the table."}
                ],
                "max_tokens": 100,
                "temperature": 1.0
            }
        }

class MountModelRequest(BaseModel):
    model_name: str = Field(default="facebook/nllb-200-distilled-600M", description="The Hugging Face model name")
    model_type: str = Field(default="translation", description="Type of model to mount. Supported model types: 'translation', 'text2text-generation', 'text-generation'.")
    source_language: Optional[str] = Field(default="eng_Latn", description="[Optional] Language code for the source language (e.g., 'eng_Latn' for English).")
    target_language: Optional[str] = Field(default="ita_Latn", description="[Optional] Language code for the target language (e.g., 'ita_Latn' for Italian).")

    class Config:
        protected_namespaces = ()  
        json_schema_extra  = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M",
                "model_type": "translation",
                "source_language": "eng_Latn",
                "target_language": "ita_Latn"
            }
        }


class TextGenerationResponse(BaseModel):
    generated_text: str = Field(
        description="The generated text from the model based on the provided prompt.",
        example="Il gatto è sul tavolo."  # Example translation output
    )

class TranslationResponse(BaseModel):
    translated_text: str = Field(
        description="The generated text from the model based on the provided prompt.",
        example="Il gatto è sul tavolo."  # Example translation output
    )    

class ModelRequest(BaseModel):
    model_name: str = Field(default="facebook/nllb-200-distilled-600M", description="The Hugging Face model name")

    class Config:
        protected_namespaces = ()  
        json_schema_extra  = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M"
            }
        }

# Global variables for tracking state
current_model = None
current_model_type = None  
tokenizer = None
translator = None  # For translation models
text_generator = None  # For text generation models
is_downloading = False
download_progress = []
port = os.getenv("PORT", "8001").strip()  # Ensure extra whitespace is removed
download_directory = os.getenv("HUGGINGFACE_CACHE_DIR", "/app/model_cache").strip()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN").strip()


@app.get("/list_models/", 
          response_model=List[Dict[str, str]], 
          summary="List downloaded models", 
          response_description="A list of models available in the cache.",
          responses={200: {"content": {"application/json": {"example": [{"model_name": "facebook/nllb-200-distilled-600M", "model_type": "translation", "model_size_bytes": "3.45 GB"}]}}}})
async def list_models() -> List[Dict[str, str]]:
    """List all downloaded models along with their types and sizes."""
    model_cache_dir = os.getenv("HUGGINGFACE_CACHE_DIR", "/app/model_cache")
    models_info = []
    try:
        # List directories in the cache directory to find downloaded models
        model_dirs = [d for d in os.listdir(model_cache_dir) if os.path.isdir(os.path.join(model_cache_dir, d))]
        for model_dir in model_dirs:
            # Skip any directories that start with '.' or are named '.locks'
            if model_dir.startswith('.') or model_dir == '.locks':
                continue
            
            model_path = os.path.join(model_cache_dir, model_dir)
            model_name = model_dir.replace('--', '/').replace("models/", "")  # Remove 'models/' and revert name
            model_type = infer_model_type(model_dir)
            model_size = get_directory_size(model_path)  # Get the total size of the model directory
            
            # Format the model size as a human-readable string
            formatted_size = format_size(model_size)
            
            models_info.append({
                "model_name": model_name,
                "model_type": model_type,
                "model_size_bytes": formatted_size  # Now it is a string
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing model cache: {str(e)}")
   
    return models_info

@app.post("/download_model/", 
          summary="Download a Model",
          description="Initiate the download of a specified model from the Hugging Face Hub. Return progress updates on the download process.")
async def download_model(request: ModelRequest) -> StreamingResponse:
    """Download specified model and stream progress."""
    global is_downloading, download_progress   
    model_name = request.model_name

    if is_downloading:
        raise HTTPException(status_code=400, detail="A download is currently in progress.")
    
    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))

    if os.path.exists(model_path):
        existing_files = os.listdir(model_path)
        if len(existing_files) >= 2:
            return {"message": f"Model '{model_name}' is already downloaded to '{model_path}'."}
    
    is_downloading = True
    download_progress = []

    # Create the download directory if it doesn't exist
    os.makedirs(download_directory, exist_ok=True)
    
    async def generate_progress():
        """Helper function to yield download progress."""
        global is_downloading, download_progress
        try:
            info = model_info(model_name)
            files = info.siblings            
            filtered_files = filter_unwanted_files(files)
            total_files = len(filtered_files)
            download_progress.append({"message": "Download started."})
          
            for index, file in enumerate(filtered_files):
                try:
                    file_path = hf_hub_download(repo_id=model_name, 
                                                filename=file.rfilename, 
                                                cache_dir=download_directory, 
                                                force_download=True,
                                                token=huggingface_token)
                    progress_update = {
                        "file_name": file.rfilename,
                        "current_index": index + 1,
                        "total_files": total_files
                    }
                    download_progress.append(progress_update)
                    yield f"data: {json.dumps(progress_update)}\n\n"
                    await asyncio.sleep(0.1)  # Simulated download delay
                except Exception as e:
                    error_message = f"Error downloading {file.rfilename}: {str(e)}"
                    download_progress.append({"error": error_message})
                    yield f"data: {json.dumps({'error': error_message})}\n\n"
                    raise
            download_progress.append({"status": "Completed"})
            move_snapshot_files(model_name, download_directory)
        finally:
            is_downloading = False
    
    return StreamingResponse(generate_progress(), media_type="text/event-stream")

@app.get("/download_progress/",
          summary="Check Download Progress",
          description="Polling method to fetch the current download progress of the model, if a download is in progress.")
async def get_progress() -> JSONResponse:
    """Returns the current download progress."""
    global download_progress, is_downloading  
    if not is_downloading:
        return JSONResponse(content={"message": "No download in progress"})
    
    return JSONResponse(content={"progress": download_progress})

@app.post("/mount_model/", 
          summary="Mount a Model", 
          description="Mount the specified model and setup the appropriate pipeline. Supported model types: 'translation', 'text2text-generation', 'text-generation'", 
          response_model=dict,
          responses={200: {"content": {"application/json": {"example": {"message": "Model 'facebook/nllb-200-distilled-600M' of type 'translation' mounted successfully."}}}}})
async def mount_model(request: MountModelRequest) -> dict:
    """Mount the specified model."""
    global current_model, tokenizer, translator, text_generator, current_model_type
    model_name = request.model_name
    requested_model_type = request.model_type

    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model path does not exist.")

    # Load model and tokenizer based on model type
    if requested_model_type in ("translation","text2text-generation"):
        current_model = SUPPORTED_MODEL_TYPES[requested_model_type].from_pretrained(model_path, token=huggingface_token)
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=huggingface_token)

        current_model_type = requested_model_type
        # Setup the translation pipeline with language parameters if provided
        if request.source_language and request.target_language:
            translator = pipeline(
                requested_model_type,
                model=current_model,
                tokenizer=tokenizer,
                src_lang=request.source_language,
                tgt_lang=request.target_language
            )
        else:
            # Setup pipeline without language parameters
            translator = pipeline(
                requested_model_type,
                model=current_model,
                tokenizer=tokenizer
            )
    
    elif requested_model_type == "text-generation":       
        current_model = SUPPORTED_MODEL_TYPES[requested_model_type].from_pretrained(model_path, token=huggingface_token)        
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=huggingface_token)

        current_model_type = requested_model_type

        text_generator = pipeline(
            requested_model_type,
            model=current_model,
            tokenizer=tokenizer,
        )

    else:
        raise HTTPException(status_code=400, detail="Unsupported model type provided.")
    

    # Move model to GPU if available
    print("attempt to load cuta")
    if torch.cuda.is_available():
        current_model.to('cuda')
        print(f"Model '{model_name}' moved to GPU.")

    return {"message": f"Model '{request.model_name}' of type '{request.model_type}' mounted successfully."}

@app.post("/unmount_model/", 
           summary="Unmount the Current Model", 
           description="Unmount the currently mounted model to free up resources.",
           response_model=dict,
           responses={200: {"content": {"application/json": {"example": {"message": "Model unmounted successfully."}}}}})
async def unmount_model() -> dict:
    """Unmounts the currently active model."""
    global tokenizer, current_model, current_model_type

    if not current_model or not tokenizer:
        raise HTTPException(status_code=400, detail="No model is currently mounted.")

    current_model, tokenizer, current_model_type = None, None, None
    return {"message": "Model unmounted successfully."}

@app.delete("/delete_model/", 
             summary="Delete Local Model", 
             description="Delete the local files of a previously mounted model based on the model name.", 
             response_model=dict,
             responses={200: {"content": {"application/json": {"example": {"message": "Model 'facebook/nllb-200-distilled-600M' has been deleted successfully."}}}}})
async def delete_model(request: ModelRequest) -> dict:
    """Delete a previously mounted model."""
    global tokenizer, current_model, current_model_type
    model_name = request.model_name
    
    # Construct the model path
    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))

    # Check if the model path exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' does not exist at path '{model_path}'.")

    # Unmount the model
    if current_model or tokenizer:
       current_model, tokenizer, current_model_type = None, None, None
       
    # Remove the model directory and its contents
    try:
        shutil.rmtree(model_path)
        return {"message": f"Model '{model_name}' has been deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the model: {str(e)}")
    

# Translation API Endpoint
@app.post("/translate/",
          summary="Translate Text",
          description="Translate input text using the mounted translation model.",
          response_model=TranslationResponse)
async def translate(translation_request: TranslationRequest) -> dict:
    global translator
    if not translator:
        raise HTTPException(status_code=400, detail="No translation model is currently mounted.")
    
    translated_text = translator(translation_request.text)

    if isinstance(translated_text, list):
        translated_text = translated_text[0]['translation_text']
    
    return TranslationResponse(translated_text=translated_text)

@app.post("/generate/",
          summary="Generate Text",
          description="Generate text based on the input prompt using the mounted text generation model.",
          response_model=TextGenerationResponse)
async def generate(text_generation_request: TextGenerationRequest) -> dict:
    if not text_generator:
        raise HTTPException(status_code=400, detail="No text generation model is currently mounted.")

    # Extracting messages directly from the request prompt
    messages = text_generation_request.prompt

    try:
        generated_results = text_generator(messages, 
                                           max_length=text_generation_request.max_tokens, 
                                           temperature=text_generation_request.temperature)
        # Log the generated results for debugging
        print("Generated results:", generated_results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

    generated_text = ''

    if isinstance(generated_results, list) and len(generated_results) > 0:
        first_result = generated_results[0]
        
        if isinstance(first_result, dict) and 'generated_text' in first_result:
            # Extract the list associated with 'generated_text'
            generated_text_list = first_result['generated_text']
            if isinstance(generated_text_list, list):
                # Find the last response where role is 'assistant'
                assistant_responses = [entry for entry in generated_text_list if entry.get('role') == 'assistant']
                
                if assistant_responses:
                    # Get the last assistant's response
                    last_assistant_response = assistant_responses[-1].get('content', '')
                    generated_text = last_assistant_response
                else:
                    raise HTTPException(status_code=500, detail="No response found with role 'assistant'.")
            else:
                raise HTTPException(status_code=500, detail="'generated_text' key does not contain a valid list.")
        else:
            raise HTTPException(status_code=500, detail="Unexpected format for generated_results.")
    else:
        raise HTTPException(status_code=500, detail="Generated results list is empty or invalid.")

    # Print the generated text for debugging
    print('generated_text: ' + generated_text)

    # Ensure that the produced text is a valid string
    if not isinstance(generated_text, str):
        raise HTTPException(status_code=500, detail="Generated text is not a valid string.")
    
    # If generated text is empty, handle it
    if generated_text.strip() == "":
        raise HTTPException(status_code=500, detail="Generated text is empty.")

    return TextGenerationResponse(generated_text=generated_text)


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
    return [file for file in files if file.rfilename not in unwanted_files]

def infer_model_type(model_dir: str) -> str:
    """Infer model type based on the 'config.json' or 'README.md' contents."""
    model_cache_dir = os.getenv("HUGGINGFACE_CACHE_DIR", "/app/model_cache")
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
                # Check if the pipeline_tag_value matches known types
                if pipeline_tag_value == "translation":
                    return "translation"
                elif pipeline_tag_value == "text-generation":
                    return "text-generation"
                elif pipeline_tag_value == "text2text-generation":
                    return "text2text-generation"

            # Search for tags specifically
            tags_match = re.search(r'tags:\s*-\s*([\w-]+(?:\n- [\w-]+)*)', readme_content)
            if tags_match:
                tags_list = tags_match.group(0).splitlines()
                # Look for specific entries in the tags list
                for tag in tags_list:
                    tag_value = tag.strip().replace('- ', '').strip()  # Clean tag value
                    if tag_value == "translation":
                        return "translation"
                    elif tag_value in ("text-generation", "chat"):
                        return "text-generation"
                    elif tag_value == "text2text-generation":
                        return "text2text-generation"              

    # Then, check in the config.json
    config_path = os.path.join(model_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            model_type = config.get("model_type", "").lower().strip()

            # If not found at the top level, check within text_config if it exists
            if not model_type and "text_config" in config:
                model_type = config["text_config"].get("model_type", "").lower().strip()

            # Infer from model_type
            if model_type in ["m2m_100", "marian", "mbart", "mistral"]:
                return "translation"
            elif model_type in ["qwen2", "t5"]:
                return "text2text-generation"
            elif model_type in ["phi3", "llama"]:
                return "text-generation"
            # Expand with other mappings based on model_type as needed.

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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(port))