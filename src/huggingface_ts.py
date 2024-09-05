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
from typing import Optional

# Load environment variables
load_dotenv()
# Suppress specific Pydantic warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

app = FastAPI(
    title="Hugging Face Transformers Service",
    description="API for downloading, mounting, and translating using machine learning models from Hugging Face",
    version="1.0.0",
)

# Supported model mappings
SUPPORTED_MODEL_TYPES = {
    'translation': AutoModelForSeq2SeqLM,
    'text-generation': AutoModelForCausalLM,
}


class TranslationRequest(BaseModel):
    text: str = Field(default="Hello, how are you?", description="The source content that should be translated")
    source_language: Optional[str] = Field(default="", description="Language code for the source language (e.g., 'en-US' for English).")
    target_language: Optional[str] = Field(default="", description="Language code for the target language (e.g., 'it_IT' for Italian).")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, how are you?",
                "source_language": "en-US",
                "target_language": "it-IT"
            }
        }

class TextGenerationRequest(BaseModel):
    prompt: str = Field(default="Translate the following from 'en-US' to 'it-IT': \nOnce upon a time in a land far away", description="The User Prompt comprises of custom instructions provided by the user, detailing the specific requirements for translation.")
    max_tokens: int = Field(default=500, description="The maximum number of tokens to generate.")
    temperature: float = Field(default=1.0, description="Sampling temperature for generation, where higher values lead to more random outputs.")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Translate the following from 'en-US' to 'it-IT': \nOnce upon a time in a land far away",
                "max_tokens": 500,
                "temperature": 1.0
            }
        }

# Input request models
class DownloadModelRequest(BaseModel):
    model_name: str = Field(default="facebook/mbart-large-50-many-to-many-mmt", description="The Hugging Face model name")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "facebook/mbart-large-50-many-to-many-mmt"
            }
        }

class MountModelRequest(BaseModel):
    model_name: str = Field(default="facebook/mbart-large-50-many-to-many-mmt", description="The Hugging Face model name")
    model_type: str = Field(default="translation", description="Type of model to mount. Supported values: 'translation', 'text-generation'.")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "facebook/mbart-large-50-many-to-many-mmt",
                "model_type": "translation"
            }
        }


# Global variables for tracking state
current_model = None
current_model_type = None  
tokenizer = None
is_downloading = False
download_progress = []
download_directory = os.getenv("HUGGINGFACE_CACHE_DIR", "/app/model_cache")

@app.post("/download_model/", 
          summary="Download a Model",
          description="Initiate the download of a specified model from the Hugging Face Hub. Return progress updates on the download process.")
async def download_model(request: DownloadModelRequest) -> StreamingResponse:
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
                    file_path = hf_hub_download(repo_id=model_name, filename=file.rfilename, cache_dir=download_directory, force_download=True)
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
          description="Mount a specified downloaded model for use. Ensure the model is downloaded beforehand.")
async def mount_model(request: MountModelRequest) -> dict:
    """Mounts a specified model."""
    global current_model, tokenizer, current_model_type
    model_name = request.model_name
    requested_model_type = request.model_type

    if requested_model_type not in SUPPORTED_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type: '{requested_model_type}'. Supported values are: {list(SUPPORTED_MODEL_TYPES.keys())}."
        )

    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model path does not exist.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    current_model = SUPPORTED_MODEL_TYPES[requested_model_type].from_pretrained(model_path)
    current_model_type = requested_model_type

    return {"message": "Model mounted successfully.", "model_type": current_model_type, "model_path": model_path}

@app.post("/unmount_model/",
          summary="Unmount the Current Model",
          description="Unmount the currently mounted model to free up resources.")
async def unmount_model() -> dict:
    """Unmounts the currently active model."""
    global tokenizer, current_model, current_model_type

    if not current_model or not tokenizer:
        raise HTTPException(status_code=400, detail="No model is currently mounted.")

    current_model, tokenizer, current_model_type = None, None, None
    return {"message": "Model unmounted successfully."}

# Translation API Endpoint
@app.post("/translate/",
          summary="Translate Text",
          description="Translate input text using the mounted translation model. Supports models that require source and target languages or those that do not.")
async def translate(translation_request: TranslationRequest) -> dict:
    global current_model, tokenizer
    if not current_model or not tokenizer:
        raise HTTPException(status_code=400, detail="No model is currently mounted.")
    
    # Check if source and target languages are provided
    if translation_request.source_language and translation_request.target_language:
        # Setup the translator with language parameters
        translator = pipeline(
            "translation",
            model=current_model,
            tokenizer=tokenizer,
            src_lang=translation_request.source_language,
            tgt_lang=translation_request.target_language
        )
    else:
        # Setup the translator without language parameters
        translator = pipeline(
            "translation",
            model=current_model,
            tokenizer=tokenizer
        )
    
    # Perform translation
    translated_text = translator(translation_request.text)

    # Extract the translated text from the response
    if isinstance(translated_text, list):
        translated_text = translated_text[0]['translation_text']

    return {"translated_text": translated_text}


# Text Generation API Endpoint
@app.post("/generate/",
          summary="Generate Text",
          description="Generate text based on the input prompt using the mounted text generation model.")
async def generate_text(text_generation_request: TextGenerationRequest) -> dict:
    """Generates text based on the input prompt using the mounted text generation model."""
    global current_model, tokenizer, current_model_type
    if not current_model or not tokenizer:
        raise HTTPException(status_code=400, detail="No model is currently mounted.")
    
    inputs = tokenizer(text_generation_request.prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = current_model.generate(
            **inputs, 
            max_length=text_generation_request.max_tokens, 
            temperature=text_generation_request.temperature
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

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
    unwanted_files = {'.gitattributes'}
    return [file for file in files if file.rfilename not in unwanted_files]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)