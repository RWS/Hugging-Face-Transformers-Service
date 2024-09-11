from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from huggingface_hub import hf_hub_download, model_info, HfApi
import torch
import os
import glob
import json
import asyncio
from dotenv import load_dotenv
import warnings
import shutil
from typing import Optional, List, Dict, Any
import re
from llama_cpp import Llama


# Load environment variables
load_dotenv()
# Suppress specific Pydantic warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

app = FastAPI(
    title="Hugging Face Transformers Service",
    description="This is a FastAPI application designed to provide an intuitive and efficient interface for working with Hugging Face models, specifically catering to translation and text generation tasks. The service allows users to **download and mount** models locally, making it possible to run model inference without requiring an internet connection once the models are downloaded.",
    version="1.0.0",
)
api = HfApi()

# Supported model mappings
SUPPORTED_MODEL_TYPES = {     
    'sequence-generation': AutoModelForSeq2SeqLM,
    'text-generation': AutoModelForCausalLM,
    'llama': Llama
}


class TranslationRequest(BaseModel):
    model_name: str = Field(description="The name of the translation model to use.")
    text: str = Field(default="The cat is on the table.", description="The source content that should be translated.")
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M",
                "text": "The cat is on the table."
            }
        }

class TextGenerationRequest(BaseModel):
    model_name: str = Field(description="The name of the text generation model to use.")
    prompt: List[Dict[str, str]] = Field(
        default=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "Please generate some text based on the following prompt."}
        ],
        description="The User Prompt comprises of custom instructions provided by the user."
    )
    max_tokens: int = Field(default=250, description="The maximum number of tokens to generate.")
    temperature: float = Field(default=1.0, description="Sampling temperature for generation.")
    result_type: str = Field(default='assistant', description="Indicates the response type: 'raw' or 'assistant'.")
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "model_name": "gpt-3",
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Generate a response to this prompt."}
                ],
                "max_tokens": 100,
                "temperature": 1.0,
                "result_type": "assistant"
            }
        }

class MountModelRequest(BaseModel):
    model_name: str = Field(default="facebook/nllb-200-distilled-600M", description="The Hugging Face model name")
    model_type: str = Field(default="translation", description="Type of model to mount. Supported model types: 'translation', 'text2text-generation', 'text-generation', 'llama'.")
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

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    model_mounted: bool  # Boolean type
    model_size_bytes: str

class LocalModel:
    def __init__(self, model_name: str, model, model_type: str, tokenizer, pipeline):
        self.model_name = model_name
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.pipeline = pipeline


# Global list to hold models
models: List[LocalModel] = []

is_downloading = False
download_progress = []
port = os.getenv("PORT", "8001").strip()  # Ensure extra whitespace is removed
download_directory = os.getenv("HUGGINGFACE_CACHE_DIR", "/app/model_cache").strip()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN").strip()


@app.get("/list_models/",
          response_model=List[ModelInfo],
          summary="List downloaded models",
          response_description="A list of models available in the cache.",
          responses={200: {"content": {"application/json": {"example": [{"model_name": "microsoft/Phi-3.5-mini-instruct", "model_mounted": "True", "model_type": "text-generation", "model_size_bytes": "7.12 GB"}]}}}})
async def list_models() -> List[ModelInfo]:
    global download_directory, models
    """List all downloaded models along with their types and sizes."""
    
    models_info = []
    try:
        # List directories in the cache directory to find downloaded models
        model_dirs = [d for d in os.listdir(download_directory) if os.path.isdir(os.path.join(download_directory, d))]
        for model_dir in model_dirs:
            # Skip any directories that start with '.' or are named '.locks'
            if model_dir.startswith('.') or model_dir == '.locks':
                continue
            
            model_path = os.path.join(download_directory, model_dir)
            model_name = model_dir.replace('--', '/').replace("models/", "")  # Remove 'models/' and revert name
            model_type = infer_model_type(model_dir)
            model_size = get_directory_size(model_path)  # Get the total size of the model directory
            
            # Format the model size as a human-readable string
            formatted_size = format_size(model_size)
            is_mounted = any(model.model_name == model_name for model in models)  # Check if model is mounted
            
            models_info.append({
                "model_name": model_name,
                "model_type": model_type,
                "model_mounted": is_mounted,
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
          description="Mount the specified model and setup the appropriate pipeline. Supported model types: 'translation', 'text2text-generation', 'text-generation', 'llama'",
          response_model=dict,
          responses={200: {"content": {"application/json": {"example": {"message": "Model 'facebook/nllb-200-distilled-600M' of type 'translation' mounted successfully."}}}}})
async def mount_model(request: MountModelRequest) -> dict:
    global models, download_directory
    """Mount the specified model."""
    
    model_name = request.model_name
    requested_model_type = request.model_type
    
    # Check if the model is already mounted
    if any(model.model_name == model_name for model in models):
        return {"message": f"Model '{model_name}' of type '{requested_model_type}' is already mounted."}

    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model path does not exist.")

    tokenizer = None
    model = None
    trans_pipeline = None
    
    # Load model and tokenizer based on the model type
    if requested_model_type in ('translation', 'text2text-generation', 'summarization'):
        model = get_model_type(requested_model_type).from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        trans_pipeline = pipeline(
            requested_model_type,
            model=model,
            tokenizer=tokenizer,
            src_lang=request.source_language,
            tgt_lang=request.target_language
        )

    elif requested_model_type == "text-generation":
        model = get_model_type(requested_model_type).from_pretrained(model_path)        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        trans_pipeline = pipeline(
            requested_model_type,
            model=model,
            tokenizer=tokenizer,
        )

    elif requested_model_type == "llama":
        gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
        if not gguf_files:
            raise HTTPException(status_code=404, detail="No .gguf file found in model directory.")
        
        filename = os.path.basename(gguf_files[0])
        model = Llama.from_pretrained(
            repo_id=model_name,
            filename=filename,
            local_dir=model_path
        )
        tokenizer = None  # Assuming LLaMA does not use a tokenizer in your context
        trans_pipeline = None

    else:
        raise HTTPException(status_code=400, detail="Unsupported model type provided.")

    # Create a LocalModel instance and add it to the models list
    local_model = LocalModel(model_name=model_name, model=model, model_type=requested_model_type, tokenizer=tokenizer, pipeline=trans_pipeline)
    models.append(local_model)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        print(f"Model '{model_name}' moved to GPU.")

    return {"message": f"Model '{request.model_name}' of type '{request.model_type}' mounted successfully."}


@app.post("/unmount_model/",
          summary="Unmount the Current Model",
          description="Unmount the currently mounted model to free up resources.",
          response_model=dict,
          responses={200: {"content": {"application/json": {"example": {"message": "Model unmounted successfully."}}}}})
async def unmount_model(request: ModelRequest) -> dict:
    global models
    """Unmounts the currently active model."""
    
    model_name = request.model_name
    
    # Check if the model is currently mounted
    model_to_unmount = next((model for model in models if model.model_name == model_name), None)
    
    if model_to_unmount is None:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not currently mounted.")

   # Free resources if necessary (e.g., if using GPU)
    if torch.cuda.is_available() and model_to_unmount.model.device.type == 'cuda':
        model_to_unmount.model.cpu()  # Move model to CPU to free GPU memory
        print(f"Model '{model_name}' moved to CPU.")

    # Remove the model from the global models list
    models.remove(model_to_unmount)

    return {"message": f"Model '{model_name}' unmounted successfully."}

@app.delete("/delete_model/",
            summary="Delete Local Model",
            description="Delete the local files of a previously mounted model based on the model name.",
            response_model=dict,
            responses={200: {"content": {"application/json": {"example": {"message": "Model 'facebook/nllb-200-distilled-600M' has been deleted successfully."}}}}})
async def delete_model(request: ModelRequest) -> dict:
    global download_directory, models
    """Delete a previously mounted model."""
    
    model_name = request.model_name
    
    # Find the model in the global models list
    model_to_delete = next((model for model in models if model.model_name == model_name), None)
    
    if model_to_delete:        
        # Free resources if it was on GPU
        if torch.cuda.is_available() and model_to_delete.model.device.type == 'cuda':
            model_to_delete.model.cpu()  # Move model to CPU to free GPU memory
            print(f"Model '{model_name}' moved to CPU.")

        # Remove the model from the global models list
        models.remove(model_to_delete)

    # Construct the model path
    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))

    # Check if the model path exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' does not exist at path '{model_path}'.")

    # Remove the model directory and its contents
    try:
        shutil.rmtree(model_path)
        return {"message": f"Model '{model_name}' has been deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the model: {str(e)}")
    

# Translation API Endpoint
@app.post("/translate/",
          summary="Translate Text",
          description="Translate input text using the specified translation model.",
          response_model=TranslationResponse)
async def translate(translation_request: TranslationRequest) -> dict:
    global models
    """Translate input text using the specified translation model."""
    
    # Find the corresponding translator model in the models list
    model_to_use = next((model for model in models if model.model_name == translation_request.model_name), None)
    if not model_to_use or not model_to_use.pipeline:
        raise HTTPException(status_code=400, detail="The specified translation model is not currently mounted.")

    translated_text = model_to_use.pipeline(translation_request.text)
    if isinstance(translated_text, list):
        translated_text = translated_text[0]['translation_text']
    
    return TranslationResponse(translated_text=translated_text)


@app.post("/generate/",
          summary="Generate Text",
          description="Generate text based on the input prompt using the specified text generation model.",
          response_model=dict)
async def generate(text_generation_request: TextGenerationRequest) -> dict:
    global models
    """Generate text using the specified text generation model."""
    
    # Find the corresponding text generation model in the models list
    model_to_use = next((model for model in models if model.model_name == text_generation_request.model_name), None)
    if not model_to_use:
        raise HTTPException(status_code=400, detail="The specified text generation model is not currently mounted.")

    messages = text_generation_request.prompt
    try:
        if model_to_use.model_type == "llama":
            generated_results = model_to_use.model.create_chat_completion(
                messages=messages,
                max_tokens=text_generation_request.max_tokens,
                temperature=text_generation_request.temperature
            )
        else:
            generated_results = model_to_use.pipeline(
                messages,
                max_length=text_generation_request.max_tokens,
                temperature=text_generation_request.temperature
            )
        print("Generated results:", generated_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

    # Determine the result_type
    result_type = text_generation_request.result_type or 'raw'  # Default to 'raw' if None or empty
    if result_type == 'raw':
        # Return the raw response
        return generated_results
    elif result_type == 'assistant':
        # Extract the assistant's response
        if model_to_use.model_type == "llama":
            assistant_response = generated_results['choices'][0]['message']['content']
        else:
            # Handle other models if needed
            if isinstance(generated_results, list) and len(generated_results) > 0:
                first_result = generated_results[0]
                if isinstance(first_result, dict) and 'generated_text' in first_result:
                    assistant_response = first_result['generated_text']
                elif isinstance(first_result, dict) and 'choices' in first_result:
                    assistant_response = first_result['choices'][0]['text']
                else:
                    assistant_response = str(first_result)
        return {"assistant_response": assistant_response}  # Return only the assistant response
    else:
        raise HTTPException(status_code=400, detail="Invalid result_type specified. Use 'raw' or 'assistant'.")
    

#@app.get("/model_info/")
async def get_model_info(model_name: str, return_type: str):
    global download_directory
    """
    Retrieve either model configuration or model information from HfApi.

    Args:
    - model_name: Name of the model to load.
    - return_type: Either 'config' to return model configuration or 'info' to return model info from HfApi.

    Returns:
    - Model configuration or model information.
    """

    # Handle invalid return_type
    if return_type not in ["config", "info"]:
        raise HTTPException(status_code=400, detail="Invalid return_type. Use 'config' or 'info'.")
    
    try:

        if return_type == "config":
            model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                model_path = os.path.join(download_directory, model_name.replace("/", "_"))  # Modify if needed
                config = AutoConfig.from_pretrained(model_name)
                
                return {
                    "model_name": model_name,
                    "config": config.to_dict()  # Convert the config object to a dictionary
                }        
            else:
                gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
                model_type = "unknown"

                if gguf_files:
                    model_type = "llama"  # or any other identifier you wish to use

                    return {
                        "model_name": model_name,
                        "message": "No configuration file found. Providing minimal information.",
                        "minimal_info": {
                            "supports_TensorFlow": hasattr(AutoModel, 'from_tf'),
                            "supports_pretrained": hasattr(AutoModel, 'from_pretrained'),
                            "model_type": model_type  # Add the detected model type
                        }
                    }
        else:
            # Using HfApi to get model info
            api = HfApi()
            model_info = api.model_info(model_name)
            return {
                "model_name": model_name,
                "info": {
                    "model_id": model_info.modelId,
                    "pipeline_tag" : model_info.pipeline_tag,
                    "transformers_info" : model_info.transformers_info,
                    "card_data" : model_info.card_data,
                    "siblings" : model_info.siblings,
                    "library_name" : model_info.library_name,
                    "widget_data" : model_info.widget_data,
                    "config" : model_info.config,
                    "spaces" : model_info.spaces,
                    "model_type": model_info.modelId.split('/')[-1],
                    "architecture": model_info.pipeline_tag,
                    "tags": model_info.tags,
                    "downloads": model_info.downloads,
                    "last_updated": model_info.lastModified,
                    "safetensors" : model_info.safetensors
                }
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
            config = json.load(f)
            model_type = config.get("model_type", "").lower().strip()

            # If not found at the top level, check within text_config if it exists
            if not model_type and "text_config" in config:
                model_type = config["text_config"].get("model_type", "").lower().strip()

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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(port))