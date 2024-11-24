from fastapi import APIRouter, Query, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import hf_hub_download, model_info
from models import ModelRequest, ModelInfo, MountModelRequest, DownloadModelRequest, LocalModel, TextGenerationRequest, TranslationRequest, GeneratedResponse
from transformers import AutoTokenizer, AutoModel, AutoConfig, pipeline
from state import model_state
#from llama_cpp import Llama
from huggingface_hub import HfApi
import glob
import os
from typing import Optional, Dict, List
from helpers import infer_model_type, get_directory_size, format_size, get_model_type, move_snapshot_files, filter_unwanted_files, extract_assistant_response
from config import config
import torch
import shutil
import json
import asyncio
from connection_manager import ConnectionManager
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import partial

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
manager = ConnectionManager()
model_states: Dict[str, Dict] = {}

# Initialize a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=5)

@router.on_event("startup")
async def startup_event():
    print(f"Server host: {config.HOST}")
    print(f"Server port: {config.PORT}")
    print(f"Models folder: {config.DOWNLOAD_DIRECTORY}")
    print(f"Device is configured to use {model_state.device}")
    print(f"Hugging Face API {config.HUGGINGFACE_TOKEN}")
    # Check if the Hugging Face token is set correctly
    if config.HUGGINGFACE_TOKEN == "" or config.HUGGINGFACE_TOKEN == "Your_Hugging_Face_API_Token":
        print("WARNING: You need to set your Hugging Face API token to download models.")
    else:
        print("Hugging Face API token is set.")

@router.on_event("shutdown")
async def shutdown_event():
    print("Shutting down the server.")


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
                            }
                        }
                    ]
                }
            }
        }
    }
)
async def list_models() -> List[ModelInfo]:
    """List all downloaded models along with their types, sizes, and properties."""
   
    models_info = []
    try:
        # List directories in the cache directory to find downloaded models
        model_dirs = [
            d for d in os.listdir(config.DOWNLOAD_DIRECTORY)
            if os.path.isdir(os.path.join(config.DOWNLOAD_DIRECTORY, d))
        ]
        for model_dir in model_dirs:
            # Skip any directories that start with '.' or are named '.locks'
            if model_dir.startswith('.') or model_dir == '.locks':
                continue
           
            model_path = os.path.join(config.DOWNLOAD_DIRECTORY, model_dir)
            model_name = model_dir.replace('--', '/').replace("models/", "")  # Remove 'models/' and revert name
            model_type = infer_model_type(model_dir)
            model_size = get_directory_size(model_path)  # Get the total size of the model directory
           
            # Format the model size as a human-readable string
            formatted_size = format_size(model_size)
            is_mounted = any(
                model.model_name == model_name for model in model_state.models
            ) 
           
            # Retrieve properties from the mounted models
            properties = {}
            if is_mounted:
                local_model = next(
                    (model for model in model_state.models if model.model_name == model_name), None
                )
                if local_model:
                    properties = local_model.properties
           
            models_info.append({
                "model_name": model_name,
                "model_type": model_type,
                "model_mounted": is_mounted,
                "model_size_bytes": formatted_size,
                "properties": properties 
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing model cache: {str(e)}")
   
    return models_info


@router.post(
    "/download_model/",
    summary="Download a Model",
    description=(
        "Initiates the download of a specified model from the Hugging Face Hub. "
        "If the model is already downloaded, it informs the user accordingly."
    ),
    response_model=dict,
    responses={
        200: {
            "description": "Successful model download initiation",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Model download started.",
                        "model_path": "C:/HuggingFace/model_cache/models--facebook/nllb-200-distilled-600M"
                    }
                }
            }
        },
        400: {
            "description": "Error in model download due to invalid request or existing download."
        }
    }
)
async def download_model_endpoint(request: DownloadModelRequest, background_tasks: BackgroundTasks) -> dict:
    client_id = request.client_id
    model_name = request.model_name
    api_key = request.api_key 

    logger.info(f"Received download request from client_id: {client_id} for model: {model_name}")

    # Check if the client has an active download
    if model_states.get(client_id, {}).get("is_downloading"):
        logger.warning(f"Client {client_id} already has an active download.")
        raise HTTPException(status_code=400, detail="A download is currently in progress for this client.")

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

    model_path = os.path.join(
        config.DOWNLOAD_DIRECTORY,
        "models--" + model_name.replace('/', '--')
    )
    if os.path.exists(model_path):
        existing_files = os.listdir(model_path)
        if len(existing_files) >= 2:
            model_states[client_id]["is_downloading"] = False
            logger.info(f"Model '{model_name}' already downloaded for client {client_id}.")
            return {
                "status": "Download not started.",
                "message": f"Model '{model_name}' is already downloaded to '{model_path}'."
            }

    # Create the download directory if it doesn't exist
    os.makedirs(config.DOWNLOAD_DIRECTORY, exist_ok=True)
    logger.info(f"Download directory ensured at {config.DOWNLOAD_DIRECTORY}")

    # Start the download in a background task
    background_tasks.add_task(download_model, client_id, model_name, api_key)

    logger.info(f"Download initiated for client {client_id}.")

    return {
        "status": "Download started.",
        "message": f"Model '{model_name}' is being downloaded."
    }

async def download_model(client_id: str, model_name: str, api_key: Optional[str]):
    try:
        logger.info(f"Starting download process for client {client_id}, model {model_name}")
        model_state = model_states[client_id]
        model_state["download_progress"]["status"] = "Download started."

        # Fetch model information in a separate thread to avoid blocking
        info = await asyncio.get_event_loop().run_in_executor(executor, lambda: model_info(model_name))
        files = info.siblings
        filtered_files = filter_unwanted_files(files)
        total_files = len(filtered_files)
        model_state["download_progress"].update({
            "status": "Downloading",
            "message": f"Downloaded 0 of {total_files}",
            "files": []
        })
        logger.info(f"Model {model_name} has {total_files} files to download.")

        for index, file in enumerate(filtered_files):
            try:
                logger.info(f"Client {client_id}: Downloading file {file.rfilename} ({index + 1}/{total_files})")
                
                token_to_use = api_key if api_key else config.HUGGINGFACE_TOKEN

                # Download the file in a separate thread with named arguments using partial
                file_path = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    partial(
                        hf_hub_download,
                        repo_id=model_name,
                        filename=file.rfilename,
                        cache_dir=config.DOWNLOAD_DIRECTORY,
                        force_download=True,
                        token=token_to_use
                    )
                )
                logger.info(f"Client {client_id}: Successfully downloaded {file.rfilename}")

                progress_update = {
                    "file_name": file.rfilename,
                    "index": index + 1,
                    "total_files": total_files
                }
                model_state["download_progress"]["files"].append(progress_update)
                model_state["download_progress"]["message"] = f"Downloaded {index + 1} of {total_files}"

                # Send progress update via WebSocket
                await manager.send_message(client_id, json.dumps({
                    "type": "file_downloaded",
                    "data": progress_update
                }))

            except Exception as e:
                error_message = f"Error downloading {file.rfilename}: {str(e)}"
                model_state["download_progress"]["error"] = error_message
                logger.error(f"Client {client_id}: {error_message}")

                # Send error message via WebSocket
                await manager.send_message(client_id, json.dumps({
                    "type": "error",
                    "data": error_message
                }))
                break  # Stop downloading on error

        if not model_state["download_progress"]["error"]:
            model_state["download_progress"].update({
                "status": "Completed",
                "message": f"Downloaded {total_files} of {total_files}"
            })
            logger.info(f"Client {client_id}: Download completed successfully.")
            await manager.send_message(client_id, json.dumps({
                "type": "completed",
                "data": f"Model '{model_name}' download completed."
            }))

            # Execute post-download tasks in a separate thread
            await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: move_snapshot_files(model_name, config.DOWNLOAD_DIRECTORY)
            )
    finally:
        model_state["is_downloading"] = False
        logger.info(f"Client {client_id}: Download state reset.")


@router.websocket("/ws/progress/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(client_id, websocket)
    logger.info(f"WebSocket connection established for client {client_id}")
    try:
        while True:
            # Keep the connection alive.
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
        logger.info(f"WebSocket connection closed for client {client_id}")





@router.post(
    "/mount_model/",
    summary="Mount a Model",
    description=(
        "Mount the specified model and setup the appropriate pipeline. "
        "Supported model types: 'translation', 'text-generation'"
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
            "description": "Unsupported model type provided.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Unsupported model type provided."
                    }
                }
            }
        },
        404: {
            "description": "Model path does not exist.",
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

    # Check if the model is already mounted
    if any(model.model_name == model_name for model in model_state.models):
        return {"message": f"Model '{model_name}' of type '{requested_model_type}' is already mounted."}

    model_path = os.path.join(config.DOWNLOAD_DIRECTORY , "models--" + model_name.replace('/', '--'))    
    if (not os.path.exists(model_path)):
        model_path = os.path.join(config.DOWNLOAD_DIRECTORY , model_name.replace('/', '--'))    
    
    if (not os.path.exists(model_path)):
        raise HTTPException(status_code=404, detail="Model path does not exist.")

    tokenizer = None
    model = None
    trans_pipeline = None    
    
    # Load model and tokenizer based on the model type
    if requested_model_type in ('translation', 'text2text-generation', 'summarization'):
            model = get_model_type(requested_model_type).from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
           
            # Dynamically add all non-null properties to pipeline_kwargs
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
        try:    
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
        except Exception as ex:
            print(f"Exception: {ex}")

    # elif requested_model_type == "llama":
    #     gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
    #     if not gguf_files:
    #         raise HTTPException(status_code=404, detail="No .gguf file found in model directory.")
        
    #     filename = os.path.basename(gguf_files[0])
    #     model = Llama.from_pretrained(
    #         repo_id=model_name,
    #         filename=filename,
    #         local_dir=model_path
    #     )
    #     tokenizer = None 
    #     trans_pipeline = None

    else:
        raise HTTPException(status_code=400, detail="Unsupported model type provided.")

    local_model = LocalModel(
        model_name=model_name,
        model=model,
        model_type=requested_model_type,
        tokenizer=tokenizer,
        pipeline=trans_pipeline,
        properties=properties 
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
    
    # Check if the model is currently mounted
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

    model_path = os.path.join(config.DOWNLOAD_DIRECTORY , "models--" + model_name.replace('/', '--'))    
    if (not os.path.exists(model_path)):
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
        
        if isinstance(translated_text, list):
            translated_text = translated_text[0]['translation_text']
    
        return GeneratedResponse(generated_response=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")


@router.post("/generate/",
          summary="Generate Text",
          description="Generate text based on the input prompt using the specified text generation model.",
          response_model=GeneratedResponse)
async def generate(text_generation_request: TextGenerationRequest) -> GeneratedResponse:
    """Generate text using the specified text generation model."""
    
    model_to_use = next((model for model in model_state.models if model.model_name == text_generation_request.model_name), None)
    if not model_to_use:
        raise HTTPException(status_code=400, detail="The specified text generation model is not currently mounted.")
    
    max_tokens = text_generation_request.max_tokens 
    if max_tokens <= 0:
        max_tokens = model_to_use.model.config.max_position_embeddings

    messages = text_generation_request.prompt
    try:
        if model_to_use.model_type == "llama":
            generated_results = model_to_use.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=text_generation_request.temperature
            )
        else:
            generated_results = model_to_use.pipeline(
                messages,
                max_length=max_tokens,
                temperature=text_generation_request.temperature
            )
        print("Generated results:", generated_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating text: {str(e)}")

    result_type = text_generation_request.result_type or 'raw'  # Default to 'raw' if None or empty
    if result_type == 'raw':
        if isinstance(generated_results, list):
            return GeneratedResponse(generated_response=generated_results)
        else:
            return GeneratedResponse(generated_response=[generated_results])
    elif result_type == 'assistant':
        assistant_response = extract_assistant_response(generated_results)
        return GeneratedResponse(generated_response=assistant_response)     
    else:
        raise HTTPException(status_code=400, detail="Invalid result_type specified. Use 'raw' or 'assistant'.")

@router.get("/model_info/",
          summary="Retrieve model info",
          description="Retrieve either model configuration or model information.",
          response_model=dict)
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
            model_path = os.path.join(config.DOWNLOAD_DIRECTORY, "models--" + model_name.replace('/', '--'))
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                model_path = os.path.join(config.DOWNLOAD_DIRECTORY, model_name.replace("/", "_"))
                config_file = AutoConfig.from_pretrained(model_name)
                
                return {
                    "model_name": model_name,
                    "config": config_file.to_dict() 
                }        
            else:
                gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
                model_type = "unknown"

                if gguf_files:
                    model_type = "llama" 

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