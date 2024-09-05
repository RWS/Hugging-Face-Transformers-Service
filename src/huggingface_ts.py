from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel,Field
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, model_info
import torch
import os
import glob
import json
import asyncio
from dotenv import load_dotenv
import warnings
import shutil  # For moving files

# Load environment variables
load_dotenv()

# Suppress specific Pydantic warnings
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

app = FastAPI()

# Supported model mappings
SUPPORTED_MODEL_TYPES = {
    'translation': AutoModelForSeq2SeqLM,
    'text-generation': AutoModelForCausalLM,
}

class DownloadModelRequest(BaseModel):
    model_name: str

    class Config:
        protected_namespaces = ()  # Disable protected namespaces 

class MountModelRequest(BaseModel):
    model_name: str
    model_type: str = Field(..., description="Type of model to mount. Supported values: 'translation', 'text-generation'.")

    class Config:
        protected_namespaces = ()  # Disable protected namespaces                     

class TranslationRequest(BaseModel):
    text: str

# Global variables for tracking state
current_model = None
current_model_type = None  # Initialize to hold the current model type
tokenizer = None
is_downloading = False
download_progress = []
download_directory = os.getenv("HUGGINGFACE_CACHE_DIR", "/app/model_cache")

@app.post("/download_model/")
async def download_model(request: DownloadModelRequest) -> StreamingResponse:
    global is_downloading, download_progress   
    model_name = request.model_name

    # Check download status
    if is_downloading:
        raise HTTPException(status_code=400, detail="A download is currently in progress.")
    
    # Construct the model path
    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))

    print(f"Local model folder: {model_path}")

    # Check if the model folder already exists and contains files
    if os.path.exists(model_path):
        existing_files = os.listdir(model_path)
        print(f"Local model folder contains '{len(existing_files)}' files")
        if len(existing_files) >= 2:  # Check if the model is "downloaded"
            return {"message": f"Model '{model_name}' is already downloaded to '{model_path}'."}    
    
    is_downloading = True
    download_progress = []  # Reset progress
    
    # Validation and setup
    os.makedirs(download_directory, exist_ok=True)
    
    async def generate_progress():
        global is_downloading, download_progress
        
        try:
            info = model_info(model_name)
            files = info.siblings
            
            # Filter out unwanted files
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
                    print(f"Current download progress: {download_progress}")  # Debugging
                    print(f"Downloaded: {file.rfilename}")
                    yield f"data: {json.dumps(progress_update)}\n\n"
                    await asyncio.sleep(0.1)  # Simulated download delay
                except Exception as e:
                    error_message = f"Error downloading {file.rfilename}: {str(e)}"
                    download_progress.append({"error": error_message})
                    yield f"data: {json.dumps({'error': error_message})}\n\n"
                    raise
            
            print("Model download completed.")
            download_progress.append({"status": "Completed"})
            # Handle snapshot folders
            move_snapshot_files(model_name, download_directory)
        
        finally:
            is_downloading = False
    
    return StreamingResponse(generate_progress(), media_type="text/event-stream")

@app.get("/download_progress/")
async def get_progress() -> JSONResponse:
    global download_progress, is_downloading  # Ensure it's using the global variable

    if not is_downloading:
        return JSONResponse(content={"message": "No download in progress"}) 
    
    print(f"Progress requested: {download_progress}")  # Debugging
    return JSONResponse(content={"progress": download_progress}) 


@app.post("/mount_model/")
async def mount_model(request: MountModelRequest) -> dict:
    global current_model, tokenizer, current_model_type
    model_name = request.model_name
    requested_model_type = request.model_type

    # Validate the requested model type
    if requested_model_type not in SUPPORTED_MODEL_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type: '{requested_model_type}'. Supported values are: {list(SUPPORTED_MODEL_TYPES.keys())}."
        )

    # Construct the model path
    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))

    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model path does not exist.")

    # Use the correct tokenizer based on requested model type
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    current_model = SUPPORTED_MODEL_TYPES[requested_model_type].from_pretrained(model_path)

    # Set the current model type globally
    current_model_type = requested_model_type  # Store the type globally

    return {"message": "Model mounted successfully.", "model_type": current_model_type, "model_path": model_path}


@app.post("/unmount_model/")
async def unmount_model() -> dict:
    global tokenizer, current_model, current_model_type
    
    if not current_model or not tokenizer:
        raise HTTPException(status_code=400, detail="No model is currently mounted.")
    
    current_model, tokenizer, current_model_type = None, None, None
    return {"message": "Model unmounted successfully."}


@app.post("/translate/")
async def translate(translation_request: TranslationRequest) -> dict:
    global current_model, tokenizer, current_model_type

    if not current_model or not tokenizer:
        raise HTTPException(status_code=400, detail="No model is currently mounted.")

    # Prepare inputs for the model
    inputs = tokenizer(translation_request.text, return_tensors="pt")

    # Generate outputs directly without checking the type
    with torch.no_grad():
        outputs = current_model.generate(**inputs)  # Directly call generate

    # Decode the output
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"translated_text": translated_text}

# Function to move snapshot files to the model path
def move_snapshot_files(model_name, download_directory):
    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))
    snapshots_path = os.path.join(model_path, "snapshots")
    
    # Check if the snapshots directory exists
    if os.path.exists(snapshots_path):
        # Clear existing files and folders in the model root, except for the snapshots folder
        for item in os.listdir(model_path):
            item_path = os.path.join(model_path, item)
            if item != 'snapshots' and os.path.exists(item_path):
                print(f"Removing {item_path} from the model root.")
                shutil.rmtree(item_path)  # Remove file or folder
        
        # Get the first snapshot directory and move its contents
        snapshot_dirs = glob.glob(os.path.join(snapshots_path, '*'))
        if snapshot_dirs:
            first_snapshot = snapshot_dirs[0]  # Only work with the first snapshot
            for item in os.listdir(first_snapshot):
                source_path = os.path.join(first_snapshot, item)
                destination_path = os.path.join(model_path, item)

                # Check if the destination already exists
                if os.path.exists(destination_path):
                    print(f"Skipping {destination_path} as it already exists.")
                    continue  # Skip the move if the file already exists
                
                # Move the file from snapshot to model_path
                shutil.move(source_path, model_path)

            # Optionally remove the snapshots directory after moving files
            shutil.rmtree(snapshots_path)

def filter_unwanted_files(files):
    unwanted_files = {'.gitattributes'}  # Set of unwanted file names
    return [file for file in files if file.rfilename not in unwanted_files]  # Filter out unwanted files

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)