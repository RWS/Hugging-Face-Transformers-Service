from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from huggingface_hub import hf_hub_download, model_info
import os
import json
import asyncio
from helpers import move_snapshot_files, filter_unwanted_files
from state import download_state # singleton instance

async def download_model(model_name: str, download_directory: str, huggingface_token: str) -> StreamingResponse:
    """Download specified model and stream progress."""  

    if download_state.is_downloading:
        raise HTTPException(status_code=400, detail="A download is currently in progress.")
    
    download_state.is_downloading = True
    download_state.download_progress = []  # Reset progress
    
    model_path = os.path.join(download_directory, "models--" + model_name.replace('/', '--'))

    if os.path.exists(model_path):
        existing_files = os.listdir(model_path)
        if len(existing_files) >= 2:
            return {"message": f"Model '{model_name}' is already downloaded to '{model_path}'."}

    # Create the download directory if it doesn't exist
    os.makedirs(download_directory, exist_ok=True)
    
    async def generate_progress():
        """Helper function to yield download progress."""    
        try:
            info = model_info(model_name)
            files = info.siblings            
            filtered_files = filter_unwanted_files(files)
            total_files = len(filtered_files)
            download_state.download_progress.append({"message": "Download started."})
          
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
                    download_state.download_progress.append(progress_update)
                    yield f"data: {json.dumps(progress_update)}\n\n"
                    await asyncio.sleep(0.1)  # Simulated download delay
                except Exception as e:
                    error_message = f"Error downloading {file.rfilename}: {str(e)}"
                    download_state.download_progress.append({"error": error_message})
                    yield f"data: {json.dumps({'error': error_message})}\n\n"
                    raise
            download_state.download_progress.append({"status": "Completed"})
            move_snapshot_files(model_name, download_directory)
        finally:
            download_state.is_downloading = False
    
    return StreamingResponse(generate_progress(), media_type="text/event-stream")