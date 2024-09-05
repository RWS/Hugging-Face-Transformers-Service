from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, model_info
import torch
import os
import glob
import json
import asyncio
from dotenv import load_dotenv
import warnings
import shutil

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

# Input request models
class DownloadModelRequest(BaseModel):
    model_name: str

    class Config:
        schema_extra = {
            "example": {
                "model_name": "facebook/translate-en-fr"
            }
        }

class MountModelRequest(BaseModel):
    model_name: str
    model_type: str = Field(..., description="Type of model to mount. Supported values: 'translation', 'text-generation'.")

    class Config:
        schema_extra = {
            "example": {
                "model_name": "facebook/translate-en-fr",
                "model_type": "translation"
            }
        }

class TranslationRequest(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "Hello, how are you?"
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

@app.post("/translate/",
          summary="Translate Text",
          description="Translate the input text using the currently mounted model.")
async def translate(translation_request: TranslationRequest) -> dict:
    """Translates input text using the mounted model."""
    global current_model, tokenizer, current_model_type
    if not current_model or not tokenizer:
        raise HTTPException(status_code=400, detail="No model is currently mounted.")
    
    inputs = tokenizer(translation_request.text, return_tensors="pt")
    with torch.no_grad():
        outputs = current_model.generate(**inputs)  
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"translated_text": translated_text}

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