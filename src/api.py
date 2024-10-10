from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import hf_hub_download, model_info
from models import ModelRequest, ModelInfo, MountModelRequest, LocalModel, TextGenerationRequest, TranslationResponse, TranslationRequest
from transformers import AutoTokenizer, AutoModel, AutoConfig, pipeline
from state import model_state  # Import the state management
from llama_cpp import Llama
from huggingface_hub import HfApi
import glob
import os
from typing import List
from helpers import infer_model_type, get_directory_size, format_size, get_model_type, move_snapshot_files, filter_unwanted_files
from config import config
import torch
import shutil
import json
import asyncio

router = APIRouter()

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


@router.get("/list_models/",
          response_model=List[ModelInfo],
          summary="List downloaded models",
          response_description="A list of models available in the cache.",
          responses={200: {"content": {"application/json": {"example": [{"model_name": "microsoft/Phi-3.5-mini-instruct", "model_mounted": "True", "model_type": "text-generation", "model_size_bytes": "7.12 GB"}]}}}})
async def list_models() -> List[ModelInfo]:
    """List all downloaded models along with their types and sizes."""
    
    models_info = []
    try:
        # List directories in the cache directory to find downloaded models
        model_dirs = [d for d in os.listdir(config.DOWNLOAD_DIRECTORY) if os.path.isdir(os.path.join(config.DOWNLOAD_DIRECTORY, d))]
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
            is_mounted = any(model.model_name == model_name for model in model_state.models)  # Check if model is mounted
            
            models_info.append({
                "model_name": model_name,
                "model_type": model_type,
                "model_mounted": is_mounted,
                "model_size_bytes": formatted_size  # Now it is a string
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error accessing model cache: {str(e)}")
   
    return models_info


@router.post("/download_model/",
             summary="Download a Model",
             description="Initiates the download of a specified model from the Hugging Face Hub. "
                         "If the model is already downloaded, it informs the user accordingly.",
             response_model=dict,
             responses={200: {
                 "description": "Successful model download",
                 "content": {
                     "application/json": {
                         "example": {
                             "message": "Model 'facebook/nllb-200-distilled-600M' is already downloaded to 'C:/HuggingFace/model_cache/models--facebook/nllb-200-distilled-600M'."
                         }
                     }
                 }},
             400: {
                 "description": "Error in model download due to invalid request or existing download."
             }})
async def download_model_endpoint(request: ModelRequest) -> dict:
    model_name = request.model_name
   
    if model_state.is_downloading:
        raise HTTPException(status_code=400, detail="A download is currently in progress.")
    
    model_state.is_downloading = True
    model_state.download_progress = {
        "status": "Starting...",
        "message": "",
        "error": None,
        "files": []
    }
    
    model_path = os.path.join(config.DOWNLOAD_DIRECTORY, "models--" + model_name.replace('/', '--'))

    if os.path.exists(model_path):
        existing_files = os.listdir(model_path)
        if len(existing_files) >= 2:
            return {"status": "Download not started.", "message": f"Model '{model_name}' is already downloaded to '{model_path}'."}


    model_state.download_progress["status"] = "Download started."

    # Create the download directory if it doesn't exist
    os.makedirs(config.DOWNLOAD_DIRECTORY, exist_ok=True)
    
    async def generate_progress():
        """Helper function to yield download progress."""    
        try:        
            info = model_info(model_name)
            files = info.siblings            
            filtered_files = filter_unwanted_files(files)
            total_files = len(filtered_files)

            model_state.download_progress["total_files"] = total_files            
            model_state.download_progress["status"] = "Downloading"
            model_state.download_progress["message"] = f"Downloaded {0} of {total_files}"

            for index, file in enumerate(filtered_files):
                try:
                    file_path = hf_hub_download(repo_id=model_name, 
                                                filename=file.rfilename, 
                                                cache_dir=config.DOWNLOAD_DIRECTORY, 
                                                force_download=True,
                                                token=config.HUGGINGFACE_TOKEN)
                    progress_update = {
                        "file_name": file.rfilename,
                        "current_index": index + 1,
                        "total_files": total_files
                    }

                    model_state.download_progress["files"].append(progress_update)
                    model_state.download_progress["message"] = f"Downloaded {index + 1} of {total_files}"
                    
                    # for clients that fully support SEE
                    yield f"data: {json.dumps(progress_update)}\n\n"
                    await asyncio.sleep(0.1)  # Simulated download delay
                except Exception as e:
                    error_message = f"Error downloading {file.rfilename}: {str(e)}"
                    model_state.download_progress["error"] = error_message

                    # for clients that fully support SEE
                    yield f"data: {json.dumps(model_state.download_progress)}\n\n"
                    raise

            model_state.download_progress["status"] = "Completed"
            move_snapshot_files(model_name, config.DOWNLOAD_DIRECTORY)
        finally:
            model_state.is_downloading = False
    
    return StreamingResponse(generate_progress(), media_type="text/event-stream")


@router.get("/download_progress/",
          summary="Check Download Progress",
          description="Polling method to fetch the current download progress of the model, if a download is in progress.")
async def get_progress() -> JSONResponse:
    """Returns the current download progress."""
    if not model_state.is_downloading:
        return JSONResponse(content={"message": "No download in progress"})
    
    return JSONResponse(content={"progress": model_state.download_progress})


@router.post("/mount_model/",
          summary="Mount a Model",
          description="Mount the specified model and setup the appropriate pipeline. Supported model types: 'translation', 'text2text-generation', 'text-generation', 'llama'",
          response_model=dict,
          responses={200: {"content": {"application/json": {"example": {"message": "Model 'facebook/nllb-200-distilled-600M' of type 'translation' mounted successfully."}}}}})
async def mount_model(request: MountModelRequest) -> dict:   
    """Mount the specified model."""
    
    model_name = request.model_name
    requested_model_type = request.model_type
    
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
        model = get_model_type(requested_model_type).from_pretrained(model_path).to(model_state.device)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        trans_pipeline = pipeline(
            requested_model_type,
            model=model,
            tokenizer=tokenizer,
            src_lang=request.source_language,
            tgt_lang=request.target_language
        )

    elif requested_model_type == "text-generation":
        model = get_model_type(requested_model_type).from_pretrained(model_path).to(model_state.device)
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
    model_state.models.append(local_model)

    return {"message": f"Model '{request.model_name}' of type '{request.model_type}' mounted successfully."}


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

    # Construct the model path
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
    

# Translation API Endpoint
@router.post("/translate/",
          summary="Translate Text",
          description="Translate input text using the specified translation model.",
          response_model=TranslationResponse)
async def translate(translation_request: TranslationRequest) -> dict:
    """Translate input text using the specified translation model."""
    
    # Find the corresponding translator model in the models list
    model_to_use = next((model for model in model_state.models if model.model_name == translation_request.model_name), None)
    if not model_to_use or not model_to_use.pipeline:
        raise HTTPException(status_code=400, detail="The specified translation model is not currently mounted.")

    input_text = translation_request.text
    try:
        translated_text = model_to_use.pipeline(input_text)
        
        if isinstance(translated_text, list):
            translated_text = translated_text[0]['translation_text']
    
        return TranslationResponse(translated_text=translated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during translation: {str(e)}")


@router.post("/generate/",
          summary="Generate Text",
          description="Generate text based on the input prompt using the specified text generation model.",
          response_model=dict)
async def generate(text_generation_request: TextGenerationRequest) -> dict:
    """Generate text using the specified text generation model."""
    
    # Find the corresponding text generation model in the models list
    model_to_use = next((model for model in model_state.models if model.model_name == text_generation_request.model_name), None)
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

    # Handle invalid return_type
    if return_type not in ["config", "info"]:
        raise HTTPException(status_code=400, detail="Invalid return_type. Use 'config' or 'info'.")
    
    try:

        if return_type == "config":
            model_path = os.path.join(config.DOWNLOAD_DIRECTORY, "models--" + model_name.replace('/', '--'))
            config_path = os.path.join(model_path, 'config.json')
            if os.path.exists(config_path):
                model_path = os.path.join(config.DOWNLOAD_DIRECTORY, model_name.replace("/", "_"))  # Modify if needed
                config_file = AutoConfig.from_pretrained(model_name)
                
                return {
                    "model_name": model_name,
                    "config": config_file.to_dict()  # Convert the config object to a dictionary
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