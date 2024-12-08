from fastapi import APIRouter, Query, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
#from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import hf_hub_url
from models import FineTuneRequest, ModelRequest, ModelInfo, MountModelRequest, DownloadModelRequest, ModelFileInfo, LocalModel, ListModelFilesRequest, ListModelFilesResponse, TextGenerationRequest, TranslationRequest, GeneratedResponse, ModelInfoResponse, DownloadDirectoryRequest, DownloadDirectoryResponse
from state import model_state
from connection_manager import ConnectionManager
from helpers import get_file_size_via_head, get_file_size_via_get, infer_model_type, get_directory_size, format_size, get_model_type, extract_assistant_response, fetch_model_info
from config import config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoConfig, Trainer, TrainingArguments, pipeline
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
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
manager = ConnectionManager()
model_states: Dict[str, Dict] = {}
executor = ThreadPoolExecutor(max_workers=5)

@router.on_event("startup")
async def startup_event():
    print(f"Server host: {config.HOST}")
    print(f"Server port: {config.PORT}")
    print(f"Models folder: {config.DOWNLOAD_DIRECTORY}")
    print(f"Device is configured to use {model_state.device}")
    # print(f"Hugging Face API {config.HUGGINGFACE_TOKEN}")
    # Check if the Hugging Face token is set correctly
    if config.HUGGINGFACE_TOKEN == "" or config.HUGGINGFACE_TOKEN == "Your_Hugging_Face_API_Token":
        print("WARNING: You need to set your Hugging Face API token to download models.")
    else:
        print("Hugging Face API token is set.")

@router.on_event("shutdown")
async def shutdown_event():
    print("Shutting down the server.")


@router.post("/download_directory", response_model=DownloadDirectoryResponse)
def get_download_path(request: DownloadDirectoryRequest):
    """
    Retrieve the current download directory.

    If `model_name` is provided and not empty, include it in the path.
    """

    download_dir = config.DOWNLOAD_DIRECTORY.replace('/', '\\')

    if not os.path.exists(download_dir):
        raise HTTPException(status_code=500, detail=f"Base download directory does not exist: {download_dir}")

    model_name = request.model_name

    if model_name and model_name.strip():    
        sanitized_model_name = model_name.replace('/', '--')
        model_path = os.path.join(download_dir, sanitized_model_name)

        return DownloadDirectoryResponse(path=model_path)
    else:
        return DownloadDirectoryResponse(path=download_dir)
    

@router.post(
    "/list_model_files/",
    summary="List Model Files with Sizes",
    description=(
        "Retrieves the list of available files in the specified Hugging Face model repository, "
        "including each file's size when available. Only files in the root directory are listed."
    ),
    response_model=ListModelFilesResponse,
    responses={
        200: {
            "description": "Successful retrieval of model files with sizes.",
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
    
    model_name = request.model_name
    api_key = request.api_key or os.getenv("HUGGINGFACE_TOKEN")
    
    if not model_name:
        raise HTTPException(status_code=400, detail="`model_name` must be provided.")
    
    logger.info(f"Received list files request for model: {model_name}")
    
    try:
        # Fetch model information using Hugging Face Hub API
        info = await asyncio.to_thread(fetch_model_info, model_name, api_key)
        
        if not hasattr(info, 'siblings'):
            raise AttributeError("ModelInfo object has no attribute 'siblings'")
        
        files_info = []

        # Extract the default branch for accurate download URLs
        # default_branch = info.sha  # Alternatively, use info.default_branch if available
        # if hasattr(info, 'default_branch') and info.default_branch:
        #     default_branch = info.default_branch
        
        for file in info.siblings:
            filename = file.rfilename
            if not filename:
                continue
            
            # **Filter Out Non-Root Files**
            if '/' in filename or '\\' in filename:
                logger.debug(f"Skipping non-root file: {filename}")
                continue
                        
            download_url = hf_hub_url(
                repo_id=model_name,
                filename=filename
                # revision=default_branch
            )
                       
            if getattr(file, 'size', None) is not None:
                formatted_size = format_size(file.size)
            else:
                size_bytes = await get_file_size_via_head(download_url)
                if size_bytes is not None:
                    formatted_size = format_size(size_bytes)
                else:
                    size_bytes = await get_file_size_via_get(download_url)
                    if size_bytes is not None:
                        formatted_size = format_size(size_bytes)
                    else:
                        formatted_size = "Unknown"
                                  
            files_info.append(
                ModelFileInfo(
                    file_name=filename,
                    file_size=formatted_size,
                    download_url=download_url
                )
            )
        
        logger.info(f"Retrieved {len(files_info)} root files from model '{model_name}'")
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
    """List all downloaded models along with their types, sizes, properties, and loaded gguf files."""
   
    models_info = []
    try:    
        model_dirs = [
            d for d in os.listdir(config.DOWNLOAD_DIRECTORY)
            if os.path.isdir(os.path.join(config.DOWNLOAD_DIRECTORY, d))
        ]
        for model_dir in model_dirs:
            # Skip any directories that start with '.' or are named '.locks'
            if model_dir.startswith('.') or model_dir == '.locks':
                continue
           
            model_path = os.path.join(config.DOWNLOAD_DIRECTORY, model_dir)
            model_name = model_dir.replace('--', '/').replace("models/", "")  # Normalize model name
            model_type = infer_model_type(model_dir, config.DOWNLOAD_DIRECTORY)
            model_size = get_directory_size(model_path)  # Calculate total size        
            formatted_size = format_size(model_size)
            is_mounted = any(
                model.model_name == model_name for model in model_state.models
            ) 
           
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
           
            file_names = None
            if model_type == 'text-generation':
                gguf_files_full = glob.glob(os.path.join(model_path, "*.[gG][gG][uU][fF]"))
                gguf_files = [os.path.basename(f) for f in gguf_files_full]
                file_names = gguf_files if gguf_files else None

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
        raise HTTPException(status_code=500, detail=f"Error accessing model cache: {str(e)}")
   
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
                    "status": "completed"
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
):
    """
    Download a file from the given URL to the destination path,
    sending progress updates via WebSockets to the client.
    """
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(download_url) as response:
                response.raise_for_status()
                total_size = response.headers.get('Content-Length')
                if total_size:
                    total_size = int(total_size)
                downloaded = 0
                # Ensure the destination directory exists
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                with open(destination_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                progress = (downloaded / total_size) * 100
                                try:
                                    await manager.send_message(
                                        client_id,
                                        json.dumps({
                                            "type": "file_progress",
                                            "data": {
                                                "file_name": filename,
                                                "progress": f"{progress:.2f}%",
                                                "downloaded": downloaded,
                                                "total": total_size
                                            }
                                        })
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to send progress update: {e}")
                                    
    except Exception as e:
        error_message = f"Failed to download {filename}: {str(e)}"
        logger.error(error_message)
        try:
            await manager.send_message(
                client_id,
                json.dumps({
                    "type": "error",
                    "data": error_message
                })
            )
        except Exception as send_err:
            logger.warning(f"Failed to send error message: {send_err}")
        raise

@router.websocket("/ws/progress/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    logger.info(f"WebSocket connection established for client {client_id}")
    try:
        while True:
            try:
                # Await and process incoming messages from the client
                data = await websocket.receive_text()
                if data == "heartbeat":
                    logger.info(f"Heartbeat received from {client_id}")
                    # Send acknowledgment
                    await manager.send_message(client_id, json.dumps({"type": "heartbeat_ack"}))
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
    
    model_name = request.model_name
    requested_model_type = request.model_type
    properties = request.properties or {}
    specified_file_name = request.file_name 

    # Check if the model is already mounted
    if any(model.model_name == model_name for model in model_state.models):
        return {"message": f"Model '{model_name}' of type '{requested_model_type}' is already mounted."}

    # Construct the model path based on the model name
    model_path = os.path.join(config.DOWNLOAD_DIRECTORY, model_name.replace('/', '--'))    
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model path does not exist.")

    tokenizer = None
    model = None
    trans_pipeline = None    

    try:
        if requested_model_type == 'translation':
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
           
            trans_pipeline = pipeline(
                requested_model_type,
                **pipeline_kwargs
            )

        elif requested_model_type == "text-generation":        
            if specified_file_name:
                # Verify that the specified *.gguf file exists and has a .gguf extension
                gguf_file_path = os.path.join(model_path, specified_file_name)
                if not os.path.exists(gguf_file_path):
                    raise HTTPException(
                        status_code=404, 
                        detail=f"The specified *.gguf file '{specified_file_name}' does not exist in the model directory."
                    )
                if not specified_file_name.lower().endswith(".gguf"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"The specified file '{specified_file_name}' does not have a '.gguf' extension."
                    )
                
                # Load the text-generation model using llama_cpp
                model = Llama.from_pretrained(
                    repo_id=model_name,
                    filename=specified_file_name,
                    local_dir=model_path,
                    verbose=False
                )
                                
                trans_pipeline = None
                tokenizer = None
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
                    print("pad_token was not set. Assigned pad_token to eos_token.")

                model = get_model_type(requested_model_type).from_pretrained(model_path, trust_remote_code=True)
                
                # Update model configuration with the pad_token_id
                model.config.pad_token_id = tokenizer.pad_token_id
                model.resize_token_embeddings(len(tokenizer))

                trans_pipeline = pipeline(
                    requested_model_type,
                    model=model,
                    torch_dtype="auto",                
                    device_map="auto",                
                    tokenizer=tokenizer)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type provided.")

    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(ex)}")

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

    response = {
        "message": f"Model '{request.model_name}' of type '{request.model_type}' mounted successfully."
    }
    if properties:
        response["properties"] = properties
    
    return response


@router.post("/unmount_model/",
          summary="Unmount the Current Model",
          description="Unmount the currently mounted model to free up resources.",
          response_model=dict,
          responses={200: {"content": {"application/json": {"example": {"message": "Model unmounted successfully."}}}}})
async def unmount_model(request: ModelRequest) -> dict:
    """Unmounts the currently active model."""
    
    model_name = request.model_name
    
    model_to_unmount = next((model for model in model_state.models if model.model_name == model_name), None)
    
    if model_to_unmount is None:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not currently mounted.")

   # Free resources if necessary (e.g., if using GPU)
    if torch.cuda.is_available() and model_to_unmount.model.device.type == 'cuda':
        model_to_unmount.model.cpu()  # Move model to CPU to free GPU memory
        print(f"Model '{model_name}' moved to CPU.")

    # Remove the model from the global models list
    model_state.models.remove(model_to_unmount)

    return {"message": f"Model '{model_name}' unmounted successfully."}

@router.delete("/delete_model/",
            summary="Delete Local Model",
            description="Delete the local files of a previously mounted model based on the model name.",
            response_model=dict,
            responses={200: {"content": {"application/json": {"example": {"message": "Model 'facebook/nllb-200-distilled-600M' has been deleted successfully."}}}}})
async def delete_model(request: ModelRequest) -> dict:
    """Delete a previously mounted model."""
    
    model_name = request.model_name
    
    # Find the model in the global models list
    model_to_delete = next((model for model in model_state.models if model.model_name == model_name), None)
    
    if model_to_delete:        
        # Free resources if it was on GPU
        if torch.cuda.is_available() and model_to_delete.model.device.type == 'cuda':
            model_to_delete.model.cpu()  # Move model to CPU to free GPU memory
            print(f"Model '{model_name}' moved to CPU.")

        # Remove the model from the global models list
        model_state.models.remove(model_to_delete)

    
    model_path = os.path.join(config.DOWNLOAD_DIRECTORY , model_name.replace('/', '--')) 

    # Check if the model path exists
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' does not exist at path '{model_path}'.")

    # Remove the model directory and its contents
    try:
        shutil.rmtree(model_path)
        return {"message": f"Model '{model_name}' has been deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the model: {str(e)}")
    

@router.post("/translate/",
          summary="Translate Text",
          description="Translate input text using the specified translation model.",
          response_model=GeneratedResponse)
async def translate(translation_request: TranslationRequest) -> GeneratedResponse:
    """Translate input text using the specified translation model."""
    
    model_to_use = next((model for model in model_state.models if model.model_name == translation_request.model_name), None)
    if not model_to_use or not model_to_use.pipeline:
        raise HTTPException(status_code=400, detail="The specified translation model is not currently mounted.")

    input_text = translation_request.text
    try:
        translated_text = model_to_use.pipeline(input_text)
        
        print("Translation result:", translated_text)

        if isinstance(translated_text, list):
            translated_text = translated_text[0]['translation_text']
    
        return GeneratedResponse(generated_response=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")


@router.post(
    "/generate/",
    summary="Generate Text",
    description="Generate text based on the input prompt using the specified text generation model.",
    response_model=GeneratedResponse
)
async def generate(text_generation_request: TextGenerationRequest) -> GeneratedResponse:
    """Generate text using the specified text generation model."""
       
    model_to_use = next(
        (model for model in model_state.models if model.model_name == text_generation_request.model_name), 
        None
    )
    if not model_to_use:
        raise HTTPException(
            status_code=400, 
            detail="The specified text generation model is not currently mounted."
        )
    
    max_tokens = text_generation_request.max_tokens
    messages = text_generation_request.prompt
    prompt_plain = text_generation_request.prompt_plain
    temperature = text_generation_request.temperature
    
    try:
        # Determine if the model uses a .gguf file based on the file_name extension
        is_llama_model = (
            model_to_use.file_name is not None and
            model_to_use.file_name.lower().endswith(".gguf")
        )
        if is_llama_model:
            # Use llama_cpp's method to generate text
            generated_results = model_to_use.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
        else:
            if max_tokens <= 0:             
                max_tokens = model_to_use.model.config.max_position_embeddings
            
            if prompt_plain:                
                generated_results = model_to_use.pipeline(
                    prompt_plain,
                    max_length=max_tokens,
                    temperature=temperature
                )
            else:
                generated_results = model_to_use.pipeline(
                    messages,
                    max_length=max_tokens,
                    temperature=temperature
                )
                
        print("Generated results:", generated_results)
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating text: {str(e)}"
        )
    
    result_type = text_generation_request.result_type or 'assistant'  # Default to 'assistant' if None or empty
    
    if result_type == 'raw':
        if isinstance(generated_results, list):
            return GeneratedResponse(generated_response=generated_results)
        else:
            return GeneratedResponse(generated_response=[generated_results])
    elif result_type == 'assistant':
        assistant_response = extract_assistant_response(generated_results)
        return GeneratedResponse(generated_response=assistant_response)    
    else:
        raise HTTPException(
            status_code=400, 
            detail="Invalid result_type specified. Use 'raw' or 'assistant'."
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
    


finetune_status = {"fine_tune": "not_started", "details": ""}


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
    # Validate that the client_id is connected
    if request.client_id not in manager.active_connections:
        raise HTTPException(status_code=400, detail=f"Client ID '{request.client_id}' is not connected via WebSocket.")
    
    # Add the fine_tune_model task to background
    background_tasks.add_task(fine_tune_model, request)
    return {"message": "Fine-tuning has started. You will receive progress updates via WebSocket.", "status": "started"}

# Define an asynchronous function to send progress messages
async def send_progress(client_id: str, message: str):
    json_message = json.dumps({"type": "progress", "message": message})
    await manager.send_message(client_id, json_message)

# Refactored fine_tune_model to handle asynchronous message sending
def fine_tune_model(request: FineTuneRequest):
    async def run():
        try:      
            client_id = request.client_id
            await send_progress(client_id, "Fine-tuning process started.")
            logger.info("Starting fine-tuning process.")

            # Validate paths
            if not Path(request.model_path).exists():
                raise FileNotFoundError(f"Model path '{request.model_path}' does not exist.")
            logger.info(f"Model path '{request.model_path}' exists.")
            await send_progress(client_id, f"Model path '{request.model_path}' exists.")

            if not Path(request.data_file).exists():
                raise FileNotFoundError(f"Data file '{request.data_file}' does not exist.")
            logger.info(f"Data file '{request.data_file}' exists.")
            await send_progress(client_id, f"Data file '{request.data_file}' exists.")

            # Optionally load validation data
            validation_dataset = None
            if request.validation_file:
                if not Path(request.validation_file).exists():
                    raise FileNotFoundError(f"Validation file '{request.validation_file}' does not exist.")
                logger.info(f"Validation file '{request.validation_file}' exists.")
                await send_progress(client_id, f"Validation file '{request.validation_file}' exists.")
                df_val = pd.read_csv(request.validation_file)
                if 'src_text' not in df_val.columns or 'tgt_text' not in df_val.columns:
                    raise ValueError("Validation CSV data file must contain 'src_text' and 'tgt_text' columns.")
                df_val['src_text'] = df_val['src_text'].astype(str).fillna('').str.strip()
                df_val['tgt_text'] = df_val['tgt_text'].astype(str).fillna('').str.strip()
                df_val = df_val[df_val['tgt_text'] != '']
                if len(df_val) == 0:
                    raise ValueError("No valid examples found in the validation dataset after filtering.")
                validation_dataset = Dataset.from_pandas(df_val)
                validation_dataset = validation_dataset.rename_column("src_text", "source")
                validation_dataset = validation_dataset.rename_column("tgt_text", "target")
                logger.info("Validation dataset loaded and prepared.")
                await send_progress(client_id, "Validation dataset loaded and prepared.")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(request.model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(request.model_path)
            logger.info("Tokenizer and Model loaded successfully.")
            await send_progress(client_id, "Tokenizer and Model loaded successfully.")

            # Detect if the model is multilingual by checking lang_code_to_id
            is_multilingual = False
            if hasattr(tokenizer, 'lang_code_to_id') and tokenizer.lang_code_to_id:
                is_multilingual = True
                logger.info("Model is detected as multilingual based on tokenizer.")
                await send_progress(client_id, "Model is detected as multilingual based on tokenizer.")

            if is_multilingual:
                if not request.source_lang or not request.target_lang:
                    raise ValueError("Source and target language codes must be provided for multilingual models.")
                if request.source_lang not in tokenizer.lang_code_to_id:
                    raise ValueError(f"Unsupported source language code: '{request.source_lang}'")
                if request.target_lang not in tokenizer.lang_code_to_id:
                    raise ValueError(f"Unsupported target language code: '{request.target_lang}'")
                logger.info(f"Source language '{request.source_lang}' and target language '{request.target_lang}' are supported.")
                await send_progress(client_id, f"Source language '{request.source_lang}' and target language '{request.target_lang}' are supported.")
            else:
                logger.info("Model is detected as non-multilingual. Language codes will be ignored.")
                await send_progress(client_id, "Model is detected as non-multilingual. Language codes will be ignored.")

            # Create output directory if it doesn't exist
            os.makedirs(request.output_dir, exist_ok=True)
            logger.info(f"Output directory '{request.output_dir}' is ready.")
            await send_progress(client_id, f"Output directory '{request.output_dir}' is ready.")

            # Load and prepare the training dataset
            df = pd.read_csv(request.data_file)
            logger.info("Training data file loaded into DataFrame.")
            await send_progress(client_id, "Training data file loaded into DataFrame.")

            if 'src_text' not in df.columns or 'tgt_text' not in df.columns:
                raise ValueError("CSV data file must contain 'src_text' and 'tgt_text' columns.")
            logger.info("Training CSV data file contains required columns.")
            await send_progress(client_id, "Training CSV data file contains required columns.")

            df['src_text'] = df['src_text'].astype(str).fillna('').str.strip()
            df['tgt_text'] = df['tgt_text'].astype(str).fillna('').str.strip()
            logger.info("Converted 'src_text' and 'tgt_text' to strings and handled missing values.")
            await send_progress(client_id, "Converted 'src_text' and 'tgt_text' to strings and handled missing values.")

            # Remove rows where 'tgt_text' is empty
            initial_count = len(df)
            df = df[df['tgt_text'] != '']
            final_count = len(df)
            logger.info(f"Filtered dataset: {final_count} out of {initial_count} examples remain.")
            await send_progress(client_id, f"Filtered dataset: {final_count} out of {initial_count} examples remain.")
            if final_count == 0:
                raise ValueError("No valid examples found in the dataset after filtering out empty 'tgt_text'.")

            # Prepare the dataset
            dataset = Dataset.from_pandas(df)
            dataset = dataset.rename_column("src_text", "source")
            dataset = dataset.rename_column("tgt_text", "target")
            logger.info("Training dataset prepared and columns renamed.")
            await send_progress(client_id, "Training dataset prepared and columns renamed.")

            # Define the preprocessing functions
            if is_multilingual:
                async def preprocess_function(examples):
                    inputs = examples["source"]
                    targets = examples["target"]

                    # Prepend target language token to the inputs to specify translation direction
                    inputs = [f">>{request.target_lang}<< {text}" for text in inputs]
                    
                    # Ensure all targets are non-empty strings
                    targets = [text if isinstance(text, str) and text.strip() else " " for text in targets]

                    # Send a sample for debugging
                    if len(inputs) > 0:
                        sample_input = inputs[0]
                        sample_target = targets[0]
                        await send_progress(client_id, f"Preprocessed Input Sample: {sample_input}")
                        await send_progress(client_id, f"Preprocessed Target Sample: {sample_target}")

                    # Set source and target languages
                    tokenizer.src_lang = request.source_lang
                    tokenizer.tgt_lang = request.target_lang
                    await send_progress(client_id, f"Set tokenizer src_lang='{request.source_lang}' and tgt_lang='{request.target_lang}'.")

                    # Tokenize the inputs and targets
                    try:
                        model_inputs = tokenizer(
                            inputs,
                            max_length=request.max_length,
                            truncation=True,
                            padding="max_length"
                        )
                        await send_progress(client_id, "Input texts tokenized.")
                    except Exception as e:
                        await send_progress(client_id, f"Error tokenizing inputs: {e}")
                        raise e

                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(
                            targets,
                            max_length=request.max_length,
                            truncation=True,
                            padding="max_length",
                        )
                    await send_progress(client_id, "Target texts tokenized.")

                    # Assign labels
                    labels_ids = labels["input_ids"]
                    model_inputs["labels"] = labels_ids
                    await send_progress(client_id, "Labels assigned to model inputs.")
                    return model_inputs

            else:
                async def preprocess_function(examples):
                    inputs = examples["source"]
                    targets = examples["target"]

                    # No language token prepending for non-multilingual models
                    
                    # Ensure all targets are non-empty strings
                    targets = [text if isinstance(text, str) and text.strip() else " " for text in targets]

                    # Send a sample for debugging
                    if len(inputs) > 0:
                        sample_input = inputs[0]
                        sample_target = targets[0]
                        await send_progress(client_id, f"Preprocessed Input Sample: {sample_input}")
                        await send_progress(client_id, f"Preprocessed Target Sample: {sample_target}")

                    # Tokenize the inputs
                    try:
                        model_inputs = tokenizer(
                            inputs,
                            max_length=request.max_length,
                            truncation=True,
                            padding="max_length"
                        )
                        await send_progress(client_id, "Input texts tokenized.")
                    except Exception as e:
                        await send_progress(client_id, f"Error tokenizing inputs: {e}")
                        raise e

                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(
                            targets,
                            max_length=request.max_length,
                            truncation=True,
                            padding="max_length",
                        )
                    await send_progress(client_id, "Target texts tokenized.")

                    # Assign labels
                    labels_ids = labels["input_ids"]
                    model_inputs["labels"] = labels_ids
                    await send_progress(client_id, "Labels assigned to model inputs.")
                    return model_inputs

            # Apply preprocessing to the training dataset
            logger.info("Applying preprocessing to the training dataset.")
            await send_progress(client_id, "Applying preprocessing to the training dataset.")
            tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['source', 'target'])
            tokenized_dataset.set_format("torch")
            logger.info("Training dataset tokenization complete.")
            await send_progress(client_id, "Training dataset tokenization complete.")

            # Optionally prepare validation dataset
            if validation_dataset:
                async def preprocess_validation(examples):
                    inputs = examples["source"]
                    targets = examples["target"]

                    if is_multilingual:
                        inputs = [f">>{request.target_lang}<< {text}" for text in inputs]
                    
                    targets = [text if isinstance(text, str) and text.strip() else " " for text in targets]

                    if is_multilingual:
                        tokenizer.src_lang = request.source_lang
                        tokenizer.tgt_lang = request.target_lang

                    try:
                        model_inputs = tokenizer(
                            inputs,
                            max_length=request.max_length,
                            truncation=True,
                            padding="max_length"
                        )
                        await send_progress(client_id, "Validation input texts tokenized.")
                    except Exception as e:
                        await send_progress(client_id, f"Error tokenizing validation inputs: {e}")
                        raise e

                    with tokenizer.as_target_tokenizer():
                        labels = tokenizer(
                            targets,
                            max_length=request.max_length,
                            truncation=True,
                            padding="max_length",
                        )
                    await send_progress(client_id, "Validation target texts tokenized.")

                    model_inputs["labels"] = labels["input_ids"]
                    await send_progress(client_id, "Validation labels assigned to model inputs.")
                    return model_inputs

                logger.info("Applying preprocessing to the validation dataset.")
                await send_progress(client_id, "Applying preprocessing to the validation dataset.")
                tokenized_validation = validation_dataset.map(preprocess_validation, batched=True, remove_columns=['source', 'target'])
                tokenized_validation.set_format("torch")
                logger.info("Validation dataset tokenization complete.")
                await send_progress(client_id, "Validation dataset tokenization complete.")
            else:
                tokenized_validation = None
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
                metric_for_best_model="loss",  # Could be customized
                fp16=torch.cuda.is_available(),
            )        
            logger.info("Training arguments set.")
            await send_progress(client_id, "Training arguments set.")

            # Initialize the Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                eval_dataset=tokenized_validation,
                tokenizer=tokenizer,
                # You can define compute_metrics if needed
            )
            logger.info("Trainer initialized.")
            await send_progress(client_id, "Trainer initialized.")
           
            logger.info("Commencing training.")
            await send_progress(client_id, "Commencing training.")
            trainer.train()
            logger.info("Training completed.")
            await send_progress(client_id, "Training completed.")
           
            trainer.save_model(request.output_dir)
            tokenizer.save_pretrained(request.output_dir)
            logger.info(f"Fine-tuned model and tokenizer saved to '{request.output_dir}'.")
            await send_progress(client_id, f"Fine-tuned model and tokenizer saved to '{request.output_dir}'.")
           
            # Final status update
            success_message = "Fine-tuning process completed successfully."
            logger.info(success_message)
            await send_progress(client_id, success_message)
        
        except Exception as e:         
            error_message = f"Error during fine-tuning: {e}"
            logger.error(error_message)
            await send_progress(request.client_id, error_message)
            # Depending on how you want to handle errors in background tasks,
            # you might log them or implement additional error handling.
    
    # Run the async function in a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run())
    loop.close()