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


class TextGenerationRequest(BaseModel):
    model_name: str = Field(..., description="The name of the text generation model to use."
    )
    prompt: Optional[List[Dict[str, str]]] = Field(
        None,
        description=(
            "The User Prompt comprised of custom instructions provided by the user.\n"
            "Should be a list of messages with roles like 'system', 'user', and 'assistant'."
        )
    )
    prompt_plain: Optional[str] = Field(
        None,
        description="A plain text prompt for models that do not use message-based inputs."
    )
    max_tokens: int = Field(
        default=250, 
        description="The maximum number of tokens to generate."
    )
    temperature: float = Field(
        default=1.0, 
        description="Sampling temperature for generation."
    )
    result_type: str = Field(
        default='assistant', 
        description="Indicates the response type: 'raw' or 'assistant'."
    )
    
    @model_validator(mode='before')
    @classmethod
    def check_prompt_types(cls, values):
        prompt, prompt_plain = values.get('prompt'), values.get('prompt_plain')
        if not prompt and not prompt_plain:
            raise ValueError("Either 'prompt' or 'prompt_plain' must be provided.")
        if prompt and prompt_plain:
            raise ValueError("Provide only one of 'prompt' or 'prompt_plain', not both.")
        return values

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "summary": "Chat-like prompt for GPT-4",
                    "value": {
                        "model_name": "gpt-4",
                        "prompt": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Generate a response to this prompt."}
                        ],
                        "max_tokens": 100,
                        "temperature": 1.0,
                        "result_type": "assistant"
                    }
                },
                {
                    "summary": "Plain text prompt for XGLM",
                    "value": {
                        "model_name": "facebook/xglm-7.5B",
                        "prompt_plain": "Translate the following English sentence to French: 'I love programming.'",
                        "max_tokens": 100,
                        "temperature": 0.7,
                        "result_type": "assistant"
                    }
                }
            ]
        }

class MountModelRequest(BaseModel):
    model_name: str = Field(...,description="The Hugging Face model name."
    )
    model_type: str = Field(...,
        description=(
            "Type of model to mount. Supported model types: "
            "'translation', 'text-generation', 'llama'."
        )
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
                "model_type": "translation",
                "properties": {
                    "src_lang": "eng_Latn",
                    "tgt_lang": "ita_Latn"
                },
                "file_name": "Marco-o1-IQ2_M.gguf"  # Example for llama models
            }
        }

class GeneratedResponse(BaseModel):
    generated_response: Union[str, List[Dict[str, Any]]] = Field(
        description=(
            "The generated text from the model based on the provided prompt. "
            "It can be either a string (when 'assistant' response is requested) "
            "or a list of dictionaries containing raw response data."
        ),
        example="Il gatto è sul tavolo.",
    )
    class Config:    
        protected_namespaces = ()      

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
    max_length: Optional[int] = Field(128, description="Maximum sequence length for tokenization.")
    save_steps: Optional[int] = Field(50, description="Number of steps between each model save.")
    save_total_limit: Optional[int] = Field(3, description="Maximum number of checkpoints to save.")  
    validation_file: Optional[str] = Field(None, description="Path to the CSV validation data file (optional).")  
    class Config:
        protected_namespaces = ()     