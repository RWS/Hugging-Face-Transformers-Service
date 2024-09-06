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
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, how are you?"
            }
        }

class TextGenerationRequest(BaseModel):
    prompt: list = Field(default="[{\"role\": \"system\",\"content\": \"you are a helpful translator\"},{\"role\": \"user\",\"content\": \"translate this from English to Italian: The cat is on the table.\"}]", description="The User Prompt comprises of custom instructions provided by the user, detailing the specific requirements for translation.")
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
    model_name: str = Field(default="facebook/nllb-200-distilled-600M", description="The Hugging Face model name")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M"
            }
        }

class MountModelRequest(BaseModel):
    model_name: str = Field(default="facebook/nllb-200-distilled-600M", description="The Hugging Face model name")
    model_type: str = Field(default="translation", description="Type of model to mount. Supported values: 'translation', 'text-generation'.")
    source_language: Optional[str] = Field(default="eng_Latn", description="[Optional] Language code for the source language (e.g., 'eng_Latn' for English).")
    target_language: Optional[str] = Field(default="ita_Latn", description="[Optional] Language code for the target language (e.g., 'ita_Latn' for Italian).")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M",
                "model_type": "translation | text-generation",
                "source_language": "eng_Latn",
                "target_language": "ita_Latn"
            }
        }

# Define a request model for deleting a model
class DeleteModelRequest(BaseModel):
    model_name: str = Field(default="facebook/nllb-200-distilled-600M", description="The Hugging Face model name")

    class Config:
        schema_extra = {
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
download_directory = os.getenv("HUGGINGFACE_CACHE_DIR", "/app/model_cache")
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

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
          description="Mount the specified model and setup the appropriate pipeline.\n The 'source_language' and 'target_language' parameters are optional for models of type 'translation'")
async def mount_model(request: MountModelRequest) -> dict:
    global current_model, tokenizer, translator, text_generator, current_model_type
    model_name = request.model_name
    requested_model_type = request.model_type

    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model path does not exist.")

    # Load model and tokenizer based on model type
    if requested_model_type == "translation":
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
          description="Unmount the currently mounted model to free up resources.")
async def unmount_model() -> dict:
    """Unmounts the currently active model."""
    global tokenizer, current_model, current_model_type

    if not current_model or not tokenizer:
        raise HTTPException(status_code=400, detail="No model is currently mounted.")

    current_model, tokenizer, current_model_type = None, None, None
    return {"message": "Model unmounted successfully."}

@app.delete("/delete_model/",
          summary="Delete Local Model",
          description="Delete the local files of a previously mounted model based on the model name.")
async def delete_model(request: DeleteModelRequest) -> dict:
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
          description="Translate input text using the mounted translation model.")
async def translate(translation_request: TranslationRequest) -> dict:
    global translator
    if not translator:
        raise HTTPException(status_code=400, detail="No translation model is currently mounted.")
    
  # Perform translation directly with the input text
    translated_text = translator(translation_request.text)

    # Extract the translated text from the response
    if isinstance(translated_text, list):
        translated_text = translated_text[0]['translation_text']
    
    return {"translated_text": translated_text}

@app.post("/generate/",
          summary="Generate Text",
          description="Generate text based on the input prompt using the mounted text generation model.")
async def generate(text_generation_request: TextGenerationRequest) -> dict:
    global text_generator

    if not text_generator:
        raise HTTPException(status_code=400, detail="No text generation model is currently mounted.")
    
    # Construct the full prompt from the structured input
    prompt_text = " ".join([msg['content'] for msg in text_generation_request.prompt])

    # Perform text generation directly with the prompt
    generated_text = text_generator(
        prompt_text,
        max_length=text_generation_request.max_tokens,
        temperature=text_generation_request.temperature
    )

    # Extract the generated text from the response
    if isinstance(generated_text, list):
        generated_text = generated_text[0]['generated_text']
    
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