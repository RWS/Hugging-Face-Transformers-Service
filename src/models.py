from pydantic import BaseModel, Field, model_validator
from typing import Union, List, Dict, Any, Optional


class TranslationRequest(BaseModel):
    model_name: str = Field(..., description="The name of the translation model to use.")
    text: str = Field(..., description="The source content that should be translated.")
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M",
                "text": "The cat is on the table."
            }
        }

class GeneratedResponse(BaseModel):
    generated_response: str
    class Config:    
        protected_namespaces = ()     


# Completion Models
class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Dict] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str
    #created: int
    model: str
    choices: List[CompletionChoice]


class CompletionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the text generation model to use.")
    prompt: str = Field(..., description="The input text prompt for text generation.")
    max_tokens: Optional[int] = Field(500, description="Maximum number of tokens to generate.")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature.")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling probability.")
    n: Optional[int] = Field(1, description="Number of completions to generate.")
    stop: Optional[List[str]] = Field(None, description="Sequences where the API will stop generating further tokens.")
    echo: Optional[bool] = Field(False, description="Whether to echo the prompt in the completion.")
    best_of: Optional[int] = Field(1, description="Generates `best_of` completions server-side and returns the best.")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "gpt-3.5-turbo",
                "prompt": "Once upon a time,",
                "max_tokens": 50,
                "temperature": 0.7,
                "top_p": 1.0,
                "n": 1,
                "stop": ["\n"],
                "echo": False,
                "best_of": 1
            }
        }


# Chat Completion Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (e.g., 'user', 'assistant').")
    content: str = Field(..., description="Content of the message.")


class ChatCompletionChoice(BaseModel):
    message: ChatMessage
    index: int
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    #created: int
    model: str
    choices: List[ChatCompletionChoice]


class ChatCompletionRequest(BaseModel):
    model_name: str = Field(..., description="Name of the chat model to use.")
    messages: List[ChatMessage] = Field(..., description="List of messages in the conversation.")
    max_tokens: Optional[int] = Field(500, description="Maximum number of tokens to generate.")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature.")
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling probability.")
    n: Optional[int] = Field(1, description="Number of completions to generate.")
    stop: Optional[List[str]] = Field(None, description="Sequences where the API will stop generating further tokens.")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Tell me a joke."}
                ],
                "max_tokens": 100,
                "temperature": 0.7,
                "top_p": 1.0,
                "n": 1,
                "stop": None
            }
        }






class MountModelRequest(BaseModel):
    model_name: str = Field(...,description="The Hugging Face model name."
    )
    properties: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional dictionary of additional properties (e.g., src_lang, tgt_lang)."
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Specific *.gguf file name to mount for 'llama' model types."
    )
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M",                
                "properties": {
                    "src_lang": "eng_Latn",
                    "tgt_lang": "ita_Latn"
                }
            }
        }





class ListModelFilesRequest(BaseModel):
    model_name: str = Field(..., description="The Hugging Face model repository name, e.g., 'username/model-name'")
    api_key: Optional[str] = Field(None, description="Your Hugging Face API token if accessing private repositories.")
    class Config:    
        protected_namespaces = ()          

class ModelFileInfo(BaseModel):
    file_name: str
    file_size: Optional[str] = None  # Human-readable file size
    download_url: Optional[str] = None  # Direct download URL
    class Config:    
        protected_namespaces = ()   

class ListModelFilesResponse(BaseModel):
    files: List[ModelFileInfo]
    class Config:    
        protected_namespaces = ()     

class DownloadModelRequest(BaseModel):
    client_id: str = Field(..., description="Unique identifier for the client")
    model_name: str = Field(..., description="The Hugging Face model repository name")
    api_key: Optional[str] = Field(None, description="Your Hugging Face API token if accessing private repositories.")
    files_to_download: Optional[List[str]] = Field(None, description="List of specific files to download. If not provided, all files will be downloaded.")
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "examples": [
                {
                    "summary": "Download only the second gguf file (Q2)",
                    "value": {
                        "client_id": "unique_client_id_123",
                        "model_name": "facebook/nllb-200-distilled-600M",
                        "files_to_download": ["pytorch_model.bin","tokenizer.json","config.json","README.md"]
                    }
                }
            ]
        }

   
class ModelRequest(BaseModel):
    model_name: str = Field(..., description="The Hugging Face model name")
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M"
            }
        }        

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    model_mounted: bool
    model_size_bytes: str
    properties: Dict[str, str]
    file_names: Optional[List[str]] = None  # lsit of *.gguf files
    loaded_file_name: Optional[str] = None  # to identify the loaded gguf file

    class Config:
        protected_namespaces = ()  

class LocalModel:
    def __init__(
        self, 
        model_name: str, 
        model, 
        model_type: str, 
        tokenizer, 
        pipeline, 
        properties: Optional[Dict[str, str]] = None,
        file_name: Optional[str] = None  
    ):
        self.model_name = model_name
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.properties = properties or {}
        self.file_name = file_name 

    class Config:
        protected_namespaces = ()

class ModelInfoResponse(BaseModel):
    model_name: str
    config: Optional[dict] = None
    message: Optional[str] = None
    minimal_info: Optional[dict] = None
    info: Optional[dict] = None

    class Config:
        protected_namespaces = ()


class DownloadDirectoryRequest(BaseModel):
    model_name: Optional[str] = None
    class Config:
        protected_namespaces = ()    

class DownloadDirectoryResponse(BaseModel):
    path: str 
    class Config:
        protected_namespaces = ()     
        
class FineTuneRequest(BaseModel):
    client_id: str = Field(..., description="Unique identifier for the client to receive progress updates.")
    model_path: str = Field(..., description="Local path to the pretrained model.")
    output_dir: str = Field(..., description="Directory where the fine-tuned model will be saved.")
    data_file: str = Field(..., description="Path to the CSV data file containing source and target texts.")
    source_lang: Optional[str] = Field(None, description="Source language code (e.g., 'en_XX'). Required for multilingual models.")
    target_lang: Optional[str] = Field(None, description="Target language code (e.g., 'it_IT'). Required for multilingual models.")
    num_train_epochs: Optional[int] = Field(4, description="Number of training epochs.")
    per_device_train_batch_size: Optional[int] = Field(2, description="Batch size per device during training.")
    per_device_eval_batch_size: Optional[int] = Field(2, description="Batch size per device during evaluation.")
    learning_rate: Optional[float] = Field(3e-5, description="Learning rate for the optimizer.")
    weight_decay: Optional[float] = Field(0.01, description="Weight decay for the optimizer.")
    max_length: Optional[int] = Field(512, description="Maximum sequence length for tokenization.")
    save_steps: Optional[int] = Field(50, description="Number of steps between each model save.")
    save_total_limit: Optional[int] = Field(3, description="Maximum number of checkpoints to save.")  
    validation_file: Optional[str] = Field(None, description="Path to the CSV validation data file (optional).")  
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "client_id": "client123",
                "model_path": "C:/HuggingFace/Models/Helsinki-NLP--opus-mt-en-it",
                "output_dir": "C:/HuggingFace/Models/Helsinki-NLP--opus-mt-en-it-finetuned-it",
                "data_file": "C:/HuggingFace/Data/data.csv",
                "validation_file": "C:/HuggingFace/Data/validation_data.csv",
                #"source_lang": "en_XX",
                #"target_lang": "it_IT",
                "num_train_epochs": 4,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "learning_rate": 3e-5,
                "weight_decay": 0.01,
                "max_length": 512,
                "save_steps": 10,
                "save_total_limit": 2                
            }
        }  