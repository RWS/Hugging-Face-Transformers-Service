from fastapi import APIRouter, Query, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from huggingface_hub import hf_hub_url
from models import FineTuneRequest, ModelRequest, ModelInfo, MountModelRequest, DownloadModelRequest, ModelFileInfo, LocalModel, ListModelFilesRequest, ListModelFilesResponse, TextGenerationRequest, TranslationRequest, GeneratedResponse, ModelInfoResponse, DownloadDirectoryRequest, DownloadDirectoryResponse
from state import model_state
from connection_manager import ConnectionManager
from helpers import get_file_size_via_head, get_file_size_via_get, infer_model_type, get_directory_size, format_size, get_model_type, extract_assistant_response, fetch_model_info
from config import config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel, AutoConfig, Trainer, TrainingArguments, pipeline, TrainerCallback, TrainerControl, TrainerState
from llama_cpp import Llama
from huggingface_hub import HfApi
import glob
import os
from typing import Optional, Dict, List
import torch
import shutil
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import aiohttp
import pandas as pd
from datasets import Dataset
import aiofiles

logger = logging.getLogger(__name__)

router = APIRouter()
manager = ConnectionManager()
model_states: Dict[str, Dict] = {}
executor = ThreadPoolExecutor(max_workers=5)

@router.on_event("startup")
async def startup_event():
    logger.info(f"Server host: {config.HOST}")
    logger.info(f"Server port: {config.PORT}")
    logger.info(f"Models folder: {config.DOWNLOAD_DIRECTORY}")
    logger.info(f"Device is configured to use {model_state.device}")
    # print(f"Hugging Face API {config.HUGGINGFACE_TOKEN}")
    # Check if the Hugging Face token is set correctly
    if config.HUGGINGFACE_TOKEN == "" or config.HUGGINGFACE_TOKEN == "Your_Hugging_Face_API_Token":
        logger.warning("WARNING: You need to set your Hugging Face API token to download models.")
    else:
        logger.info("Hugging Face API token is set.")

@router.on_event("shutdown")
async def shutdown_event():
    print("Shutting down the server.")


@router.post("/download_directory", response_model=DownloadDirectoryResponse)
def get_download_path(request: DownloadDirectoryRequest) -> DownloadDirectoryResponse:
    """
    Retrieve the current download directory.
    If `model_name` is provided and not empty, include it in the path.
    """
    download_dir = config.DOWNLOAD_DIRECTORY.replace('/', '\\')
    logger.info('POST /download_directory/ - Retrieving download directory path.')
    
    if not os.path.exists(download_dir):
        logger.error(f"Base download directory does not exist: {download_dir}")
        raise HTTPException(status_code=500, detail=f"Base download directory does not exist: {download_dir}")
    
    model_name = request.model_name
    if model_name and model_name.strip():
        sanitized_model_name = model_name.replace('/', '--')
        model_path = os.path.join(download_dir, sanitized_model_name)
        logger.info(f"Download Directory with model_name '{model_name}': '{model_path}'")
        return DownloadDirectoryResponse(path=model_path)
    else:
        logger.info(f"Download Directory without specific model: '{download_dir}'")
        return DownloadDirectoryResponse(path=download_dir)
    

@router.post(
    "/list_model_files/",
    summary="List Model Files",
    description=(
        "Retrieves the list of available files in the specified Hugging Face model repository, "
        "including each file's size when available. Only files in the root directory are listed."
    ),
    response_model=ListModelFilesResponse,
    responses={
        200: {
            "description": "Successful retrieval of model files.",
            "content": {
                "application/json": {
                    "example": {
                        "files": [
                            {"file_name": "config.json", "file_size": "15.60 KB", "download_url": "https://..."},
                            {"file_name": "pytorch_model.bin", "file_size": "350.45 MB", "download_url": "https://..."},
                        ]
                    }
                }
            },
        },
        404: {
            "description": "Model not found."
        },
        400: {
            "description": "Error due to invalid request or issues fetching model information."
        },
    },
)
async def list_model_files_endpoint(
    request: ListModelFilesRequest
) -> ListModelFilesResponse:
    """
    Retrieves the list of available files in the specified Hugging Face model repository,
    including each file's size when available. Only files in the root directory are listed.
    """
    logger.info('POST /list_model_files/ - Listing model files.')
    model_name = request.model_name
    api_key = request.api_key or os.getenv("HUGGINGFACE_TOKEN")
    
    if not model_name:
        logger.warning("list_model_files_endpoint called without 'model_name'.")
        raise HTTPException(status_code=400, detail="`model_name` must be provided.")
    
    logger.info(f"Received list files request for model: '{model_name}'")
    
    try:
        # Fetch model information using Hugging Face Hub API
        logger.info(f"Fetching model info for '{model_name}' using Hugging Face API.")
        info = await asyncio.to_thread(fetch_model_info, model_name, api_key)
        
        if not hasattr(info, 'siblings'):
            error_msg = "ModelInfo object has no attribute 'siblings'"
            logger.error(f"{error_msg} for model '{model_name}'")
            raise AttributeError(error_msg)
        
        files_info = []
        logger.info(f"Processing sibling files for model '{model_name}'.")
        
        for file in info.siblings:
            filename = file.rfilename
            if not filename:
                logger.info("Skipping file with no filename.")
                continue
            
            # **Filter Out Non-Root Files**
            if '/' in filename or '\\' in filename:
                logger.info(f"Skipping non-root file: '{filename}'")
                continue
            
            download_url = hf_hub_url(
                repo_id=model_name,
                filename=filename
                # revision=default_branch  # Uncomment if you use revision
            )
            
            if getattr(file, 'size', None) is not None:
                formatted_size = format_size(file.size)
                logger.info(f"File '{filename}' has size {formatted_size}.")
            else:
                size_bytes = await get_file_size_via_head(download_url, api_key)
                if size_bytes is not None:
                    formatted_size = format_size(size_bytes)
                else:
                    size_bytes = await get_file_size_via_get(download_url, api_key)
                    if size_bytes is not None:
                        formatted_size = format_size(size_bytes)
                    else:
                        formatted_size = "Unknown"
                        logger.warning(f"Could not determine size for file '{filename}'.")
            
            files_info.append(
                ModelFileInfo(
                    file_name=filename,
                    file_size=formatted_size,
                    download_url=download_url
                )
            )
        
        logger.info(f"Retrieved {len(files_info)} root files from model '{model_name}'.")
        return ListModelFilesResponse(files=files_info)
    
    except AttributeError as ae:
        error_message = f"Error accessing model files: {str(ae)}"
        logger.error(f"AttributeError: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    except Exception as e:
        error_message = f"Error fetching files for model '{model_name}': {str(e)}"
        logger.error(f"Exception: {error_message}")
        raise HTTPException(status_code=400, detail=error_message)
    
@router.get(
    "/list_models/",
    response_model=List[ModelInfo],
    summary="List downloaded models",
    response_description="A list of models available in the cache.",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": [
                        {
                            "model_name": "microsoft/Phi-3.5-mini-instruct",
                            "model_type": "text-generation",
                            "model_mounted": True,
                            "model_size_bytes": "7.12 GB",
                            "properties": {
                                "src_lang": "eng_Latn",
                                "tgt_lang": "ita_Latn"
                            },
                            "file_names": None,
                            "loaded_file_name": None  # No gguf file loaded
                        },
                        {
                            "model_name": "bartowski/Marco-o1-GGUF",
                            "model_type": "text-generation",
                            "model_mounted": True,
                            "model_size_bytes": "15.3 GB",
                            "properties": {},
                            "file_names": [
                                "Marco-o1-IQ2_M.gguf",
                                "Marco-o1-Another.gguf"
                            ],
                            "loaded_file_name": "Marco-o1-IQ2_M.gguf" 
                        }
                    ]
                }
            }
        }
    }
)
async def list_models() -> List[ModelInfo]:
    """List all downloaded models along with their types, sizes, properties, and mounted files."""
    
    logger.info('GET /list_models/ - Listing all downloaded models.')
    models_info = []
    try:
        model_dirs = [
            d for d in os.listdir(config.DOWNLOAD_DIRECTORY)
            if os.path.isdir(os.path.join(config.DOWNLOAD_DIRECTORY, d))
        ]
        logger.info(f"Found {len(model_dirs)} directories in download path '{config.DOWNLOAD_DIRECTORY}'.")
        
        for model_dir in model_dirs:
            # Skip any directories that start with '.' or are named '.locks'
            if model_dir.startswith('.') or model_dir == '.locks':
                logger.info(f"Skipping directory '{model_dir}' based on naming convention.")
                continue
            
            model_path = os.path.join(config.DOWNLOAD_DIRECTORY, model_dir)
            model_name = model_dir.replace('--', '/').replace("models/", "")  # Normalize model name

            model_type = infer_model_type(model_path)
            model_size = get_directory_size(model_path)  # Calculate total size        
            formatted_size = format_size(model_size)
            is_mounted = any(
                model.model_name == model_name for model in model_state.models
            )
            
            logger.info(f"Processing model '{model_name}': Type='{model_type}', Size='{formatted_size}', Mounted={is_mounted}")
            
            # Retrieve properties from the mounted models
            properties = {}
            loaded_file_name = None
            if is_mounted:
                local_model = next(
                    (model for model in model_state.models if model.model_name == model_name), None
                )
                if local_model:
                    properties = local_model.properties
                    loaded_file_name = local_model.file_name  # the loaded gguf file name
                    logger.info(f"Model '{model_name}' is mounted with properties: {properties} and loaded_file_name: {loaded_file_name}")
            
            file_names = None
            if model_type == 'text-generation':
                gguf_files_full = glob.glob(os.path.join(model_path, "*.[gG][gG][uU][fF]"))
                gguf_files = [os.path.basename(f) for f in gguf_files_full]
                file_names = gguf_files if gguf_files else None
                if file_names:
                    logger.info(f"Found gguf files for model '{model_name}': {file_names}")
                else:
                    logger.info(f"No gguf files found for model '{model_name}'.")
            
            models_info.append(ModelInfo(
                model_name=model_name,
                model_type=model_type,
                model_mounted=is_mounted,
                model_size_bytes=formatted_size,
                properties=properties,
                file_names=file_names,
                loaded_file_name=loaded_file_name  # Include the loaded gguf file name
            ))
    except Exception as e:
        logger.error(f"Error accessing model cache: {e}")
        raise HTTPException(status_code=500, detail=f"Error accessing model cache: {str(e)}")
    
    logger.info(f"Successfully listed {len(models_info)} models.")
    return models_info



@router.post(
    "/download_model/",
    summary="Download a Model",
    description=(
        "Initiates the download of a specified model from the Hugging Face Hub. "
        "If the model is already downloaded, it overwrites existing files. "
        "Users can specify which files to download. "
        "If no files are specified, all files from the model will be downloaded."
    ),
    response_model=dict,
    responses={
        200: {
            "description": "Successful model download initiation",
            "content": {
                "application/json": {
                    "example": {
                        "status": "Download started.",
                        "message": "Model 'model_name' is being downloaded."
                    }
                }
            }
        },
        400: {
            "description": "Error in model download due to invalid request or existing download."
        }
    }
)
async def download_model_endpoint(
    request: DownloadModelRequest,
    background_tasks: BackgroundTasks
) -> dict:
    client_id = request.client_id
    model_name = request.model_name
    api_key = request.api_key if request.api_key else config.HUGGINGFACE_TOKEN 
    files_to_download = request.files_to_download
   
    logger.info('POST /download_model/')

    logger.info(
        f"Received download request from client_id: {client_id} "
        f"for model: {model_name} with files: {files_to_download}"
    )
    # Check if the client has an active download
    if model_states.get(client_id, {}).get("is_downloading"):
        logger.warning(f"Client {client_id} already has an active download.")
        raise HTTPException(
            status_code=400,
            detail="A download is currently in progress for this client."
        )
    # Initialize the model state for this client
    model_states[client_id] = {
        "is_downloading": True,
        "cancel_requested": False,
        "download_progress": {
            "status": "Starting...",
            "message": "",
            "error": None,
            "files": []
        }
    }
    # Create the download directory if it doesn't exist
    os.makedirs(config.DOWNLOAD_DIRECTORY, exist_ok=True)
    logger.info(f"Download directory ensured at {config.DOWNLOAD_DIRECTORY}")
    # Start the download in a background task, passing the list of files
    background_tasks.add_task(
        download_model,
        client_id,
        model_name,
        api_key,
        files_to_download
    )
    logger.info(f"Download initiated for client {client_id}.")
    return {
        "status": "Download started.",
        "message": f"Model '{model_name}' is being downloaded."
    }

async def download_model(
    client_id: str,
    model_name: str,
    api_key: Optional[str],
    files_to_download: Optional[List[str]] = None
):
    try:
        logger.info(f"Starting download process for client {client_id}, model {model_name}")
        model_state = model_states[client_id]
        model_state["download_progress"]["status"] = "Download started."
                    
        info = await asyncio.to_thread(fetch_model_info, model_name, api_key=api_key)
        
        available_files = [file.rfilename for file in info.siblings]
        
        if files_to_download:
            invalid_files = [f for f in files_to_download if f not in available_files]
            if invalid_files:
                error_message = f"The following files are not available in the model repository: {invalid_files}"
                model_state["download_progress"]["error"] = error_message
                logger.error(f"Client {client_id}: {error_message}")
                await manager.send_message(
                    client_id,
                    json.dumps({
                        "type": "error",
                        "data": error_message
                    })
                )
                return
            filtered_files = files_to_download
            logger.info(f"Client {client_id}: {len(filtered_files)} files selected for download.")
        else:
            # If files_to_download is None or empty, download all files
            filtered_files = available_files
            logger.info(
                f"Client {client_id}: No specific files requested. Downloading all {len(filtered_files)} files."
            )
        
        total_files = len(filtered_files)
        if total_files == 0:
            error_message = "No files available to download."
            model_state["download_progress"]["error"] = error_message
            logger.error(f"Client {client_id}: {error_message}")
            await manager.send_message(
                client_id,
                json.dumps({
                    "type": "error",
                    "data": error_message
                })
            )
            return
        
        model_state["download_progress"].update({
            "status": "Downloading",
            "message": "Downloading started.",
            "files": [],
            "overall_progress": "0.00%"
        })
        logger.info(f"Model {model_name} has {total_files} files to download.")
        
        # Determine the default branch
        #default_branch = info.default_branch if hasattr(info, 'default_branch') and info.default_branch else info.sha
        
        # Calculate total size for overall progress
        total_size = 0
        file_sizes = {}
        for file in info.siblings:
            if file.rfilename in filtered_files:
                if getattr(file, 'size', None) is not None:
                    file_sizes[file.rfilename] = file.size
                    total_size += file.size
                else:
                    file_sizes[file.rfilename] = None 
        
        downloaded_total = 0
        
        for index, filename in enumerate(filtered_files):
            
            # check if cancel requested
            if model_state.get("cancel_requested"):
                logger.info(f"Client {client_id}: Cancellation requested. Stopping download.")
                model_state["download_progress"]["status"] = "Cancelled"
                model_state["download_progress"]["message"] = "Download was cancelled by the user."
                await manager.send_message(
                    client_id,
                    json.dumps({
                        "type": "cancelled",
                        "data": "Download was cancelled."
                    })
                )               
                return 
                       
            try:
                start_message = {
                    "file_name": filename,
                    "index": index + 1,
                    "total_files": total_files,
                    "status": "started"
                }
                await manager.send_message(
                    client_id,
                    json.dumps({
                        "type": "file_started",
                        "data": start_message
                    })
                )
                                
                logger.info(
                    f"Client {client_id}: Downloading file {filename} ({index + 1}/{total_files})"
                )
                                               
                download_url = hf_hub_url(
                    repo_id=model_name,
                    filename=filename
                    #revision=default_branch
                )
                
                destination_path = os.path.join(config.DOWNLOAD_DIRECTORY, model_name.replace('/', '--'), filename)                
                # Start downloading the file with progress tracking
                await download_file(
                    download_url=download_url,
                    destination_path=destination_path,
                    client_id=client_id,
                    filename=filename,
                    token=api_key if api_key else config.HUGGINGFACE_TOKEN
                )
                
                logger.info(
                    f"Client {client_id}: Successfully downloaded {filename}"
                )
                
                progress_update = {
                    "file_name": filename,
                    "index": index + 1,
                    "total_files": total_files,
                    "status": "file_completed"
                }
                model_state["download_progress"]["files"].append(progress_update)
                model_state["download_progress"]["message"] = (
                    f"Downloaded {index + 1} of {total_files} files."
                )
                
                await manager.send_message(
                    client_id,
                    json.dumps({
                        "type": "file_completed",
                        "data": progress_update
                    })
                )
                
                # Update overall progress
                file_size = file_sizes.get(filename)
                if file_size:
                    downloaded_total += file_size
                    overall_progress = (downloaded_total / total_size) * 100 if total_size else 0
                    model_state["download_progress"]["overall_progress"] = f"{overall_progress:.2f}%"
                    
                    await manager.send_message(
                        client_id,
                        json.dumps({
                            "type": "overall_progress",
                            "data": {
                                "progress": f"{overall_progress:.2f}%"
                            }
                        })
                    )
                
                # check if cancel requested
                if model_state.get("cancel_requested"):
                    logger.info(f"Client {client_id}: Cancellation requested during file download.")
                    model_state["download_progress"]["status"] = "Cancelled"
                    model_state["download_progress"]["message"] = "Download was cancelled by the user."
                    await manager.send_message(
                        client_id,
                        json.dumps({
                            "type": "cancelled",
                            "data": "Download was cancelled."
                        })
                    )
                    # Clean up partial downloads if necessary
                    if os.path.exists(destination_path):
                        try:
                            os.remove(destination_path)
                            logger.info(f"Removed incomplete file: {destination_path}")
                        except Exception as remove_err:
                            logger.warning(f"Failed to remove incomplete file: {remove_err}")
                    return
                                
            except Exception as e:
                error_message = f"Error downloading {filename}: {str(e)}"
                model_state["download_progress"]["error"] = error_message
                logger.error(f"Client {client_id}: {error_message}")
                await manager.send_message(
                    client_id,
                    json.dumps({
                        "type": "error",
                        "data": error_message
                    })
                )
                break  # Stop downloading on error
        
        if not model_state["download_progress"]["error"]:
            model_state["download_progress"].update({
                "status": "Completed",
                "message": f"Downloaded {total_files} of {total_files} files.",
                "overall_progress": "100.00%"
            })
            logger.info(f"Client {client_id}: Download completed successfully.")
            await manager.send_message(
                client_id,
                json.dumps({
                    "type": "completed",
                    "data": f"Model '{model_name}' download completed."
                })
            )
            #await asyncio.to_thread(move_snapshot_files, model_name, config.DOWNLOAD_DIRECTORY)
    finally:
        model_state["is_downloading"] = False
        logger.info(f"Client {client_id}: Download state reset.")

async def download_file(
    download_url: str,
    destination_path: str,
    client_id: str,
    filename: str,
    token: Optional[str] = None,
    chunk_size: int = 5 * 1024 * 1024  # 5MB
    #chunk_size: int = 8 * 1024  # 8KB
):
    """
    Download a file from the given URL to the destination path,
    sending progress updates via WebSockets to the client.

    Args:
        download_url (str): The URL to download the file from.
        destination_path (str): The local path to save the downloaded file.
        client_id (str): The identifier for the WebSocket client.
        filename (str): The name of the file being downloaded.
        token (Optional[str]): Authorization token, if required.
        chunk_size (int): The size of each chunk to read from the response.
    """
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(download_url) as response:
                response.raise_for_status()
                total_size = response.headers.get('Content-Length')
                if total_size is not None:
                    total_size = int(total_size)
                downloaded = 0

                # Ensure the destination directory exists
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)

                async with aiofiles.open(destination_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        if chunk:
                            
                            # Check if cancel was requested
                            if model_states.get(client_id, {}).get("cancel_requested"):
                                logger.info(f"Client {client_id}: Cancellation requested. Aborting file download.")
                                await manager.send_message(
                                    client_id,
                                    json.dumps({
                                        "type": "cancelled",
                                        "data": f"Download of {filename} was cancelled."
                                    })
                                )
                                # Clean up partial download
                                await f.close()
                                if os.path.exists(destination_path):
                                    try:
                                        os.remove(destination_path)
                                        logger.info(f"Removed incomplete file: {destination_path}")
                                    except Exception as remove_err:
                                        logger.warning(f"Failed to remove incomplete file: {remove_err}")
                                return  # Exit the download_file function


                            await f.write(chunk)
                            downloaded += len(chunk)

                            if total_size:
                                progress = (downloaded / total_size) * 100
                                progress_data = {
                                    "type": "file_progress",
                                    "data": {
                                        "file_name": filename,
                                        "progress": f"{progress:.2f}%",
                                        "downloaded": downloaded,
                                        "total": total_size
                                    }
                                }
                                try:
                                    await manager.send_message(client_id, json.dumps(progress_data))
                                except Exception as e:
                                    logger.warning(f"Failed to send progress update: {e}")
        # After successful download, notify the client
        completion_message = {
            "type": "file_completed",
            "data": f"Download of {filename} completed successfully."
        }
        await manager.send_message(client_id, json.dumps(completion_message))
        logger.info(f"Download of {filename} completed successfully for client {client_id}.")
    except Exception as e:
        error_message = f"Failed to download {filename}: {str(e)}"
        logger.error(error_message)
        error_data = {
            "type": "error",
            "data": error_message
        }
        try:
            await manager.send_message(client_id, json.dumps(error_data))
        except Exception as send_err:
            logger.warning(f"Failed to send error message: {send_err}")
        
        # Clean up partial downloads
        if os.path.exists(destination_path):
            try:
                os.remove(destination_path)
                logger.info(f"Removed incomplete file: {destination_path}")
            except Exception as remove_err:
                logger.warning(f"Failed to remove incomplete file: {remove_err}")
        raise

@router.websocket("/ws/progress/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    logger.info(f"WebSocket connection established for client {client_id}")
    model_states[client_id] = {"cancel_requested": False}
    try:
        while True:
            try:
                # Await and process incoming messages from the client
                data = await websocket.receive_text()
                if data == "heartbeat":
                    logger.info(f"Heartbeat received from {client_id}")
                    # Send acknowledgment
                    await manager.send_message(client_id, json.dumps({"type": "heartbeat_ack"}))
                elif data == "cancel":
                    logger.info(f"Cancellation request received from {client_id}")                    
                    model_states[client_id]["cancel_requested"] = True
                    await manager.send_message(
                        client_id,
                        json.dumps({
                            "type": "cancellation_ack",
                            "data": "Cancellation requested. Stopping process..."
                        })
                    )              
                else:
                    logger.info(f"Received message from {client_id}: {data}")
            except WebSocketDisconnect:
                logger.info(f"WebSocketDisconnect: Client {client_id} disconnected.")
                await manager.disconnect(client_id)
                break
            except Exception as e:
                logger.error(f"Error in WebSocket communication with {client_id}: {e}")
                await manager.disconnect(client_id)
                break
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket endpoint for {client_id}: {e}")
        await manager.disconnect(client_id)
    finally:
        model_states.pop(client_id, None)


@router.post(
    "/mount_model/",
    summary="Mount a Model",
    description=(
        "Mount the specified model and setup the appropriate pipeline. "
        "Supported model types: 'translation', 'text-generation'."
    ),
    response_model=dict,
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "message": "Model 'facebook/nllb-200-distilled-600M' of type 'translation' mounted successfully.",
                        "properties": {
                            "src_lang": "eng_Latn",
                            "tgt_lang": "ita_Latn"
                        }
                    }
                }
            }
        },
        400: {
            "description": "Unsupported model type provided or invalid file_name for 'text-generation' model.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Unsupported model type provided."
                    }
                }
            }
        },
        404: {
            "description": "Model path or specified gguf file does not exist.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model path does not exist."
                    }
                }
            }
        }
    }
)
async def mount_model(request: MountModelRequest) -> dict:
    """Mount the specified model."""
    
    logger.info("POST /mount_model/ - Mount model request received.")
    model_name = request.model_name
    properties = request.properties or {}
    specified_file_name = request.file_name

    # Check if the model is already mounted
    if any(model.model_name == model_name for model in model_state.models):
        logger.warning(f"Model '{model_name}' is already mounted.")
        return {"message": f"Model '{model_name}' is already mounted."}
    
    # Construct the model path based on the model name
    model_path = os.path.join(config.DOWNLOAD_DIRECTORY, model_name.replace('/', '--'))
    logger.info(f"Resolved model path: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Model path '{model_path}' does not exist.")
        raise HTTPException(status_code=404, detail="Model path does not exist.")
    
    requested_model_type = infer_model_type(model_path)

    logger.info(
        f"Mount Model Parameters - Model Name: {model_name}, Type: {requested_model_type}, "
        f"Properties: {properties}, Specified File Name: {specified_file_name}"
    )

    tokenizer = None
    model = None
    trans_pipeline = None

    try:
        if requested_model_type == 'translation':
            logger.info(f"Loading translation model '{model_name}' from '{model_path}'.")
            model = get_model_type(requested_model_type).from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # Prepare pipeline keyword arguments
            pipeline_kwargs = {
                "model": model,
                "tokenizer": tokenizer
            }
            for key, value in properties.items():
                if value is not None:
                    pipeline_kwargs[key] = value
                    logger.info(f"Added pipeline property - {key}: {value}")

            trans_pipeline = pipeline(
                requested_model_type,
                **pipeline_kwargs
            )
            logger.info(f"Translation pipeline for model '{model_name}' has been set up successfully.")

        elif requested_model_type == "text-generation":
            logger.info(f"Preparing to load text-generation model '{model_name}'.")
            if specified_file_name:
                # Verify that the specified *.gguf file exists and has a .gguf extension
                gguf_file_path = os.path.join(model_path, specified_file_name)
                logger.info(f"Specified gguf file path: {gguf_file_path}")

                if not os.path.exists(gguf_file_path):
                    logger.error(f"The specified *.gguf file '{specified_file_name}' does not exist in the model directory.")
                    raise HTTPException(
                        status_code=404,
                        detail=f"The specified *.gguf file '{specified_file_name}' does not exist in the model directory."
                    )
                if not specified_file_name.lower().endswith(".gguf"):
                    logger.error(f"The specified file '{specified_file_name}' does not have a '.gguf' extension.")
                    raise HTTPException(
                        status_code=400,
                        detail=f"The specified file '{specified_file_name}' does not have a '.gguf' extension."
                    )
                
                # Load the text-generation model using llama_cpp
                logger.info(f"Loading text-generation model '{model_name}' with gguf file '{specified_file_name}'.")
                model = Llama.from_pretrained(
                    repo_id=model_name,
                    filename=specified_file_name,
                    local_dir=model_path,
                    verbose=False
                )
                logger.info(f"Text-generation model '{model_name}' loaded successfully with gguf file.")

                trans_pipeline = None
                tokenizer = None
            else:
                logger.info(f"Loading text-generation model '{model_name}' without specifying a gguf file.")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
                    logger.warning("pad_token was not set. Assigned pad_token to eos_token.")
                
                model = get_model_type(requested_model_type).from_pretrained(model_path, trust_remote_code=True)
                
                # Update model configuration with the pad_token_id
                model.config.pad_token_id = tokenizer.pad_token_id
                logger.info(f"Set model's pad_token_id to {tokenizer.pad_token_id}.")
                model.resize_token_embeddings(len(tokenizer))
                logger.info(f"Resized token embeddings to match tokenizer length: {len(tokenizer)}.")

                trans_pipeline = pipeline(
                    requested_model_type,
                    model=model,
                    torch_dtype="auto",
                    device_map="auto",
                    tokenizer=tokenizer
                )
                logger.info(f"Text-generation pipeline for model '{model_name}' has been set up successfully.")
        else:
            logger.error(f"Unsupported model type provided: '{requested_model_type}'.")
            raise HTTPException(status_code=400, detail="Unsupported model type provided.")
    
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions to be handled by FastAPI
        logger.error(f"HTTPException during model mounting: {http_exc.detail}")
        raise http_exc
    except Exception as ex:
        logger.error(f"Error loading model '{model_name}': {str(ex)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(ex)}")
    
    # Create and store the local model instance
    local_model = LocalModel(
        model_name=model_name,
        model=model,
        model_type=requested_model_type,
        tokenizer=tokenizer,
        pipeline=trans_pipeline,
        properties=properties,
        file_name=specified_file_name if specified_file_name else None
    )
    model_state.models.append(local_model)
    logger.info(f"Model '{model_name}' of type '{requested_model_type}' mounted successfully.")
    
    response = {
        "message": f"Model '{request.model_name}' of type '{requested_model_type}' mounted successfully."
    }
    if properties:
        response["properties"] = properties
        logger.info(f"Mount model response properties: {properties}")
    
    return response


@router.post("/unmount_model/",
          summary="Unmount Model",
          description="Unmount the mounted model to free up resources.",
          response_model=dict,
          responses={200: {"content": {"application/json": {"example": {"message": "Model unmounted successfully."}}}}})
async def unmount_model(request: ModelRequest) -> dict:
    """Unmounts the model."""
    
    logger.info("POST /unmount_model/ - Unmount model request received.")
    model_name = request.model_name
    logger.info(f"Model to unmount: {model_name}")

    model_to_unmount = next((model for model in model_state.models if model.model_name == model_name), None)
    
    if model_to_unmount is None:
        logger.warning(f"Attempted to unmount non-existent model '{model_name}'.")
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not currently mounted.")
    
    try:
        # Free resources if necessary (e.g., if using GPU)
        if torch.cuda.is_available() and hasattr(model_to_unmount.model, 'device') and model_to_unmount.model.device.type == 'cuda':
            model_to_unmount.model.cpu()  # Move model to CPU to free GPU memory
            logger.info(f"Model '{model_name}' moved to CPU to free GPU memory.")
        
        # Remove the model from the global models list
        model_state.models.remove(model_to_unmount)
        logger.info(f"Model '{model_name}' has been unmounted and removed from the global models list.")
    
    except Exception as e:
        logger.error(f"Error unmounting model '{model_name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error unmounting model: {str(e)}")
    
    return {"message": f"Model '{model_name}' unmounted successfully."}

@router.delete("/delete_model/",
            summary="Delete Local Model",
            description="Delete the local files of a previously mounted model based on the model name.",
            response_model=dict,
            responses={200: {"content": {"application/json": {"example": {"message": "Model 'facebook/nllb-200-distilled-600M' has been deleted successfully."}}}}})
async def delete_model(request: ModelRequest) -> dict:
    """Delete a previously mounted model."""
    
    logger.info("DELETE /delete_model/ - Initiated deletion process.")
    model_name = request.model_name

    # Find the model in the global models list
    model_to_delete = next((model for model in model_state.models if model.model_name == model_name), None)

    if model_to_delete:
        logger.info(f"Model '{model_name}' found. Proceeding with deletion.")
        # Free resources if it was on GPU
        if torch.cuda.is_available() and model_to_delete.model.device.type == 'cuda':
            model_to_delete.model.cpu()  # Move model to CPU to free GPU memory
            logger.info(f"Model '{model_name}' moved to CPU to free GPU memory.")
        # Remove the model from the global models list
        model_state.models.remove(model_to_delete)
        logger.info(f"Model '{model_name}' removed from the global models list.")
    else:
        logger.warning(f"Model '{model_name}' not found in the global models list.")

    model_path = os.path.join(config.DOWNLOAD_DIRECTORY, model_name.replace('/', '--'))
    logger.info(f"Resolved model path: {model_path}")

    # Check if the model path exists
    if not os.path.exists(model_path):
        logger.error(f"Model path '{model_path}' does not exist for model '{model_name}'.")
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' does not exist at path '{model_path}'."
        )

    # Remove the model directory and its contents
    try:
        shutil.rmtree(model_path)
        logger.info(f"Model directory '{model_path}' and its contents have been deleted successfully.")
        return {"message": f"Model '{model_name}' has been deleted successfully."}
    except Exception as e:
        logger.error(f"An error occurred while deleting the model '{model_name}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while deleting the model: {str(e)}"
        )
    

@router.post("/translate/",
          summary="Translate Text",
          description="Translate input text using the specified translation model.",
          response_model=GeneratedResponse)
async def translate(translation_request: TranslationRequest) -> GeneratedResponse:
    """Translate input text using the specified translation model."""
    
    logger.info("POST /translate/ - Translation request received.")
    model_to_use = next(
        (model for model in model_state.models if model.model_name == translation_request.model_name), None
    )
    
    if not model_to_use or not model_to_use.pipeline:
        logger.warning(f"Translation model '{translation_request.model_name}' is not mounted or pipeline is unavailable.")
        raise HTTPException(
            status_code=400,
            detail="The specified translation model is not currently mounted."
        )
    
    input_text = translation_request.text
    logger.info(f"Input text for translation: {input_text}")

    try:
        translated_text = model_to_use.pipeline(input_text)
        logger.info(f"Translation result for model '{translation_request.model_name}': {translated_text}")

        if isinstance(translated_text, list):
            if translated_text:
                translated_text = translated_text[0].get('translation_text', '')
                logger.info(f"Extracted translated text: {translated_text}")
            else:
                logger.warning("Translation pipeline returned an empty list.")
                translated_text = ""

        return GeneratedResponse(generated_response=translated_text)
    except Exception as e:
        logger.error(f"Error during translation with model '{translation_request.model_name}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during translation: {str(e)}"
        )


@router.post(
    "/generate/",
    summary="Generate Text",
    description="Generate text based on the input prompt using the specified text generation model.",
    response_model=GeneratedResponse
)
async def generate(text_generation_request: TextGenerationRequest) -> GeneratedResponse:
    """Generate text using the specified text generation model."""
    
    logger.info("POST /generate/ - Text generation request received.")
    model_to_use = next(
        (
            model for model in model_state.models
            if model.model_name == text_generation_request.model_name
        ),
        None
    )
    
    if not model_to_use:
        logger.warning(f"Text generation model '{text_generation_request.model_name}' is not mounted.")
        raise HTTPException(
            status_code=400,
            detail="The specified text generation model is not currently mounted."
        )
    
    max_tokens = text_generation_request.max_tokens
    messages = text_generation_request.prompt
    prompt_plain = text_generation_request.prompt_plain
    temperature = text_generation_request.temperature

    logger.info(
        f"Generation parameters - Model: {text_generation_request.model_name}, "
        f"Max Tokens: {max_tokens}, Temperature: {temperature}, "
        f"Prompt Plain: {prompt_plain}, Messages: {messages}"
    )
    
    try:
        # Determine if the model uses a .gguf file based on the file_name extension
        is_llama_model = (
            model_to_use.file_name is not None and
            model_to_use.file_name.lower().endswith(".gguf")
        )
        logger.info(f"Is LLaMA model (.gguf): {is_llama_model}")

        if is_llama_model:
            # Use llama_cpp's method to generate text
            generated_results = model_to_use.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            logger.info(f"Generated results using LLaMA model '{text_generation_request.model_name}': {generated_results}")
        else:
            if max_tokens <= 0:
                max_tokens = model_to_use.model.config.max_position_embeddings
                logger.info(
                    f"Max tokens <= 0. Set to model's max_position_embeddings: {max_tokens}"
                )
            
            if prompt_plain:
                generated_results = model_to_use.pipeline(
                    prompt_plain,
                    max_length=max_tokens,
                    temperature=temperature
                )
                logger.info(f"Generated results with plain prompt: {generated_results}")
            else:
                generated_results = model_to_use.pipeline(
                    messages,
                    max_length=max_tokens,
                    temperature=temperature
                )
                logger.info(f"Generated results with messages: {generated_results}")
        
        # Log the generated results
        logger.info(f"Generated results: {generated_results}")
    except Exception as e:
        logger.error(f"Error generating text with model '{text_generation_request.model_name}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text: {str(e)}"
        )
    
    result_type = text_generation_request.result_type or 'assistant'  # Default to 'assistant' if None or empty
    logger.info(f"Result type requested: {result_type}")

    try:
        if result_type == 'raw':
            if isinstance(generated_results, list):
                logger.info("Returning raw generated results as list.")
                return GeneratedResponse(generated_response=generated_results)
            else:
                logger.info("Wrapping single generated result into a list.")
                return GeneratedResponse(generated_response=[generated_results])
        elif result_type == 'assistant':
            assistant_response = extract_assistant_response(generated_results)
            logger.info(f"Assistant response extracted: {assistant_response}")
            return GeneratedResponse(generated_response=assistant_response)
        else:
            logger.warning(f"Invalid result_type specified: {result_type}")
            raise HTTPException(
                status_code=400,
                detail="Invalid result_type specified. Use 'raw' or 'assistant'."
            )
    except Exception as e:
        logger.error(f"Error processing generated results: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing generated results: {str(e)}"
        )     


@router.get(
    "/model_info/",
    summary="Retrieve model info",
    description="Retrieve either model configuration or model information.",
    response_model=ModelInfoResponse
)
async def get_model_info(
    model_name: str = Query(..., description="The name of the Hugging Face model"),
    return_type: str = Query(default="info", description="Specify 'config' to retrieve model configuration or 'info' to retrieve model information.")
):
    """
    Retrieve either model configuration or model information from HfApi.
    
    Args:
    - model_name: Name of the model to load.
    - return_type: Either 'config' to return model configuration or 'info' to return model info from HfApi.
    
    Returns:
    - Model configuration or model information.
    """

    logger.info('GET /model_info/ - request received.')

    if return_type not in ["config", "info"]:
        raise HTTPException(status_code=400, detail="Invalid return_type. Use 'config' or 'info'.")

    try:
        if return_type == "config": 
            model_path = os.path.join(config.DOWNLOAD_DIRECTORY , model_name.replace('/', '--'))     

            # First, look for 'config.json' in the model directory
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                # Load configuration from the local path
                config_file = AutoConfig.from_pretrained(model_path)
                
                return ModelInfoResponse(
                    model_name=model_name,
                    config=config_file.to_dict()
                )
            else:
                # If 'config.json' is not found, check for '*.gguf' files                
                gguf_pattern = os.path.join(model_path, "*.[gG][gG][uU][fF]")
                gguf_files_full = glob.glob(gguf_pattern)
                
                # Extract only the base file names
                gguf_files = [os.path.basename(f) for f in gguf_files_full]
                
                model_type = "unknown"
                if gguf_files:
                    model_type = "text-generation"
                
                if model_type == "text-generation" and gguf_files:
                    return ModelInfoResponse(
                        model_name=model_name,
                        message="No configuration file found. Providing minimal information.",
                        minimal_info={
                            "supports_TensorFlow": hasattr(AutoModel, 'from_tf'),
                            "supports_pretrained": hasattr(AutoModel, 'from_pretrained'),
                            "model_type": model_type,
                            "file_names": gguf_files
                        }
                    )
                else:
                    # If no 'config.json' and no '*.gguf' files, return unknown type
                    return ModelInfoResponse(
                        model_name=model_name,
                        message="No configuration file or '*.gguf' files found. Providing minimal information.",
                        minimal_info={
                            "supports_TensorFlow": hasattr(AutoModel, 'from_tf'),
                            "supports_pretrained": hasattr(AutoModel, 'from_pretrained'),
                            "model_type": model_type,
                            "file_names": gguf_files if gguf_files else []
                        }
                    )
        
        else: 
            api = HfApi()
            model_info = api.model_info(model_name)
            
            info_dict = {
                "model_id": model_info.modelId,
                "pipeline_tag": model_info.pipeline_tag,
                "transformers_info": model_info.transformers_info,
                "card_data": model_info.card_data,
                "siblings": model_info.siblings,
                "library_name": model_info.library_name,
                "widget_data": model_info.widget_data,
                "config": model_info.config,
                "spaces": model_info.spaces,
                "architecture": model_info.pipeline_tag,
                "tags": model_info.tags,
                "downloads": model_info.downloads,
                "last_updated": model_info.lastModified,
                "safetensors": model_info.safetensors
            }
        
            model_type = model_info.pipeline_tag or "unknown"

            return ModelInfoResponse(
                model_name=model_name,
                info=info_dict
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

   
    
@router.post("/fine-tune/", summary="Fine-Tune a Pretrained Translation Model",
             description="Fine-tunes a pretrained translation model with provided parameters and sends real-time progress updates via WebSocket.")
async def fine_tune(
    request: FineTuneRequest,
    background_tasks: BackgroundTasks
):
    """
    Initiates the fine-tuning of a pretrained translation model based on the provided parameters.
    The fine-tuning process runs in the background and sends progress updates via WebSocket.
    """
    logger.info('POST /fine-tune/ - request received.')

    # Validate that the client_id is connected
    if request.client_id not in manager.active_connections:
        logger.warning(f"Client ID '{request.client_id}' is not connected via WebSocket.")
        raise HTTPException(
            status_code=400,
            detail=f"Client ID '{request.client_id}' is not connected via WebSocket."
        )

    # Initialize the cancellation state
    model_states[request.client_id] = {"cancel_requested": False}

    # Add the fine_tune_model task to background
    background_tasks.add_task(fine_tune_model, request)
    logger.info(f"Fine-tuning task added to background for client {request.client_id}")
    return {
        "message": "Fine-tuning has started. You will receive progress updates via WebSocket.",
        "status": "started"
    }
async def fine_tune_model(request: FineTuneRequest):
    client_id = request.client_id
    queue = asyncio.Queue()
    progress_task = asyncio.create_task(progress_sender(client_id, queue))

    try:
        await send_progress(client_id, "Fine-tuning process started.")
        logger.info("Starting fine-tuning process.")

        model_type = infer_model_type(request.model_path)
        if model_type == "unknown":
            raise ValueError("Unable to determine the model type. Supported types are 'translation' and 'text-generation'.")

        logger.info(f"Inferred model type: {model_type}")
        await send_progress(client_id, f"Inferred model type: {model_type}")

        # Validate paths
        model_path = request.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory '{model_path}' does not exist.")
        logger.info(f"Model directory '{model_path}' exists.")
        await send_progress(client_id, f"Model directory '{model_path}' exists.")

        if not os.path.exists(request.data_file):
            raise FileNotFoundError(f"Data file '{request.data_file}' does not exist.")
        logger.info(f"Data file '{request.data_file}' exists.")
        await send_progress(client_id, f"Data file '{request.data_file}' exists.")

        # Data Validation based on model type
        expected_columns = {}
        if model_type == "translation":
            expected_columns = {"src_text", "tgt_text"}
        elif model_type == "text-generation":
            expected_columns = {"prompt", "completion"}
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        df = pd.read_csv(request.data_file)
        logger.info("Data file loaded into DataFrame.")
        await send_progress(client_id, "Data file loaded into DataFrame.")

        # Check for required columns
        if not expected_columns.issubset(df.columns):
            missing = expected_columns - set(df.columns)
            raise ValueError(f"Data file is missing required columns: {missing}")

        logger.info(f"Data file contains required columns: {expected_columns}")
        await send_progress(client_id, f"Data file contains required columns: {expected_columns}")

        # Further preprocessing based on model type
        if model_type == "translation":
            df = df.rename(columns={"src_text": "source", "tgt_text": "target"})
        elif model_type == "text-generation":
            df = df.rename(columns={"prompt": "source", "completion": "target"})

        # Remove rows with empty target
        initial_count = len(df)
        df = df[df['target'].notna() & (df['target'].str.strip() != '')]
        final_count = len(df)
        logger.info(f"Filtered dataset: {final_count} out of {initial_count} examples remain.")
        await send_progress(client_id, f"Filtered dataset: {final_count} out of {initial_count} examples remain.")
        if final_count == 0:
            raise ValueError("No valid examples found in the dataset after filtering out empty 'target'.")

        # Prepare the dataset
        dataset = Dataset.from_pandas(df)
        logger.info("Dataset prepared and columns renamed.")
        await send_progress(client_id, "Dataset prepared and columns renamed.")

        # Optionally load validation data
        validation_dataset = None
        if request.validation_file:
            if not os.path.exists(request.validation_file):
                raise FileNotFoundError(f"Validation file '{request.validation_file}' does not exist.")
            logger.info(f"Validation file '{request.validation_file}' exists.")
            await send_progress(client_id, f"Validation file '{request.validation_file}' exists.")

            df_val = pd.read_csv(request.validation_file)

            # Check for required columns in validation data
            if not expected_columns.issubset(df_val.columns):
                missing = expected_columns - set(df_val.columns)
                raise ValueError(f"Validation data file is missing required columns: {missing}")

            if model_type == "translation":
                df_val = df_val.rename(columns={"src_text": "source", "tgt_text": "target"})
            elif model_type == "text-generation":
                df_val = df_val.rename(columns={"prompt": "source", "completion": "target"})

            # Remove empty targets
            df_val = df_val[df_val['target'].notna() & (df_val['target'].str.strip() != '')]
            if len(df_val) == 0:
                raise ValueError("No valid examples found in the validation dataset after filtering out empty 'target'.")

            validation_dataset = Dataset.from_pandas(df_val)
            logger.info("Validation dataset loaded and prepared.")
            await send_progress(client_id, "Validation dataset loaded and prepared.")

        # Load tokenizer and model based on model type
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if model_type == "translation":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        elif model_type == "text-generation":
            model = AutoModelForCausalLM.from_pretrained(model_path)
        logger.info("Tokenizer and Model loaded successfully.")
        await send_progress(client_id, "Tokenizer and Model loaded successfully.")

        # Handle multilingual models if applicable
        is_multilingual = False
        if model_type == "translation" and hasattr(tokenizer, 'lang_code_to_id') and tokenizer.lang_code_to_id:
            is_multilingual = True
            logger.info("Model is detected as multilingual based on tokenizer.")
            await send_progress(client_id, "Model is detected as multilingual based on tokenizer.")

            if not request.source_lang or not request.target_lang:
                raise ValueError("Source and target language codes must be provided for multilingual models.")
            if request.source_lang not in tokenizer.lang_code_to_id:
                raise ValueError(f"Unsupported source language code: '{request.source_lang}'")
            if request.target_lang not in tokenizer.lang_code_to_id:
                raise ValueError(f"Unsupported target language code: '{request.target_lang}'")
            logger.info(f"Source language '{request.source_lang}' and target language '{request.target_lang}' are supported.")
            await send_progress(client_id, f"Source language '{request.source_lang}' and target language '{request.target_lang}' are supported.")

        else:
            logger.info("Model is detected as non-multilingual or not applicable. Language codes will be ignored.")
            await send_progress(client_id, "Model is detected as non-multilingual or not applicable. Language codes will be ignored.")

        # Create output directory if it doesn't exist
        os.makedirs(request.output_dir, exist_ok=True)
        logger.info(f"Output directory '{request.output_dir}' is ready.")
        await send_progress(client_id, f"Output directory '{request.output_dir}' is ready.")

        # Define the preprocessing function based on model type
        def preprocess_function(examples):
            inputs = examples["source"]
            targets = examples["target"]
            if is_multilingual and model_type == "translation":
                inputs = [f">>{request.target_lang}<< {text}" for text in inputs]
            targets = [
                text if isinstance(text, str) and text.strip() else " "
                for text in targets
            ]
            if len(inputs) > 0:
                sample_input = inputs[0]
                sample_target = targets[0]
                queue.put_nowait(f"Preprocessed Input Sample: {sample_input}")
                queue.put_nowait(f"Preprocessed Target Sample: {sample_target}")
            if is_multilingual and model_type == "translation":
                tokenizer.src_lang = request.source_lang
                tokenizer.tgt_lang = request.target_lang
                queue.put_nowait(f"Set tokenizer src_lang='{request.source_lang}' and tgt_lang='{request.target_lang}'.")

            try:
                model_inputs = tokenizer(
                    inputs,
                    max_length=request.max_length,
                    truncation=True,
                    padding="max_length",
                )
                queue.put_nowait("Input texts tokenized.")
            except Exception as e:
                queue.put_nowait(f"Error tokenizing inputs: {e}")
                raise e

            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=request.max_length,
                    truncation=True,
                    padding="max_length",
                )
            queue.put_nowait("Target texts tokenized.")
            model_inputs["labels"] = labels["input_ids"]
            queue.put_nowait("Labels assigned to model inputs.")
            return model_inputs

        # Apply preprocessing to the training dataset
        logger.info("Applying preprocessing to the training dataset.")
        await send_progress(client_id, "Applying preprocessing to the training dataset.")
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=["source", "target"],
            desc="Tokenizing the dataset",
            num_proc=4,
        )
        tokenized_dataset.set_format("torch")
        logger.info("Training dataset tokenization complete.")
        await send_progress(client_id, "Training dataset tokenization complete.")

        # Optionally prepare validation dataset
        tokenized_validation = None
        if validation_dataset:
            def preprocess_validation(examples):
                inputs = examples["source"]
                targets = examples["target"]
                if is_multilingual and model_type == "translation":
                    inputs = [f">>{request.target_lang}<< {text}" for text in inputs]
                targets = [
                    text if isinstance(text, str) and text.strip() else " "
                    for text in targets
                ]
                if is_multilingual and model_type == "translation":
                    tokenizer.src_lang = request.source_lang
                    tokenizer.tgt_lang = request.target_lang
                    queue.put_nowait(f"Set tokenizer src_lang='{request.source_lang}' and tgt_lang='{request.target_lang}'.")

                try:
                    model_inputs = tokenizer(
                        inputs,
                        max_length=request.max_length,
                        truncation=True,
                        padding="max_length",
                    )
                    queue.put_nowait("Validation input texts tokenized.")
                except Exception as e:
                    queue.put_nowait(f"Error tokenizing validation inputs: {e}")
                    raise e
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        targets,
                        max_length=request.max_length,
                        truncation=True,
                        padding="max_length",
                    )
                queue.put_nowait("Validation target texts tokenized.")
                model_inputs["labels"] = labels["input_ids"]
                queue.put_nowait("Validation labels assigned to model inputs.")
                return model_inputs

            logger.info("Applying preprocessing to the validation dataset.")
            await send_progress(client_id, "Applying preprocessing to the validation dataset.")
            tokenized_validation = validation_dataset.map(
                preprocess_validation,
                batched=True,
                remove_columns=["source", "target"],
                desc="Tokenizing the validation dataset",
                num_proc=4,
            )
            tokenized_validation.set_format("torch")
            logger.info("Validation dataset tokenization complete.")
            await send_progress(client_id, "Validation dataset tokenization complete.")
        else:
            logger.info("No validation dataset provided.")
            await send_progress(client_id, "No validation dataset provided.")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=request.output_dir,
            num_train_epochs=request.num_train_epochs,
            per_device_train_batch_size=request.per_device_train_batch_size,
            per_device_eval_batch_size=request.per_device_eval_batch_size,
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            logging_dir=os.path.join(request.output_dir, "logs"),
            logging_steps=10,
            save_steps=request.save_steps,
            save_total_limit=request.save_total_limit,
            save_strategy="steps",
            evaluation_strategy="epoch" if tokenized_validation else "no",
            eval_steps=request.save_steps if tokenized_validation else None,
            load_best_model_at_end=True if tokenized_validation else False,
            metric_for_best_model="loss",
            fp16=torch.cuda.is_available(),
        )
        logger.info("Training arguments set.")
        await send_progress(client_id, "Training arguments set.")

        # Initialize callbacks
        websocket_progress_callback = WebSocketProgressCallback(client_id, queue)
        cancel_training_callback = CancelTrainingCallback(client_id)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_validation,
            tokenizer=tokenizer,
            callbacks=[websocket_progress_callback, cancel_training_callback],
        )
        logger.info("Trainer initialized.")
        await send_progress(client_id, "Trainer initialized.")
        
        logger.info("Commencing training.")
        await send_progress(client_id, "Commencing training.")

        # Run trainer.train() in a separate thread to avoid blocking the event loop        
        train_future = asyncio.to_thread(trainer.train)
        await train_future

        # Check if cancellation was requested
        if model_states.get(client_id, {}).get("cancel_requested", False):
            logger.info("Training was cancelled by the user.")
            await send_progress(client_id, "Training was cancelled.")
            return

        logger.info("Training completed.")
        await send_progress(client_id, "Training completed.")

        # Save the fine-tuned model and tokenizer
        trainer.save_model(request.output_dir)
        tokenizer.save_pretrained(request.output_dir)
        logger.info(f"Fine-tuned model and tokenizer saved to '{request.output_dir}'.")
        await send_progress(client_id, f"Fine-tuned model and tokenizer saved to '{request.output_dir}'.")

        success_message = "Fine-tuning process completed successfully."
        logger.info(success_message)
        await send_progress(client_id, success_message)

    except Exception as e:
        error_message = f"Error during fine-tuning: {e}"
        logger.error(error_message)
        await send_progress(client_id, error_message)
    finally:
        await queue.put("TRAINING_DONE")
        await progress_task
        # Clean up the cancellation request state
        model_states.pop(client_id, None)

async def progress_sender(client_id: str, queue: asyncio.Queue):
    while True:
        message = await queue.get()
        if message == "TRAINING_DONE":
            break
        await send_progress(client_id, message)
        queue.task_done()

async def send_progress(client_id: str, message: str):
    json_message = json.dumps({"type": "progress", "message": message})
    await manager.send_message(client_id, json_message)

class WebSocketProgressCallback(TrainerCallback):
    def __init__(self, client_id: str, queue: asyncio.Queue):
        self.client_id = client_id
        self.queue = queue

    def on_train_begin(self, args, state, control, **kwargs):
        self.queue.put_nowait("Training started.")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.queue.put_nowait(f"Epoch {state.epoch} started.")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.queue.put_nowait(f"Epoch {state.epoch} ended.")

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            self.queue.put_nowait(f"Step {state.global_step} started.")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            self.queue.put_nowait(f"Step {state.global_step} completed.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            message = f"Epoch {state.epoch} | Step {state.global_step}: " + ", ".join([f"{k}: {v}" for k, v in logs.items()])
            self.queue.put_nowait(message)

class CancelTrainingCallback(TrainerCallback):
    def __init__(self, client_id: str):
        self.client_id = client_id

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if model_states.get(self.client_id, {}).get("cancel_requested", False):
            logger.info(f"Cancellation detected for client {self.client_id}. Stopping training.")
            asyncio.create_task(send_progress(
                self.client_id,
                "Cancellation requested. Stopping training..."
            ))
            return TrainerControl.STOP
        return control

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if model_states.get(self.client_id, {}).get("cancel_requested", False):
            logger.info(f"Cancellation detected for client {self.client_id} at epoch end. Stopping training.")
            asyncio.create_task(send_progress(
                self.client_id,
                "Cancellation requested. Stopping training..."
            ))
            return TrainerControl.STOP
        return control        