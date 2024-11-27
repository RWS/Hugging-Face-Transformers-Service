from pydantic import BaseModel, Field
from typing import Union, List, Dict, Any, Optional


class TranslationRequest(BaseModel):
    model_name: str = Field(description="The name of the translation model to use.")
    text: str = Field(default="The cat is on the table.", description="The source content that should be translated.")
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M",
                "text": "The cat is on the table."
            }
        }

class TextGenerationRequest(BaseModel):
    model_name: str = Field(description="The name of the text generation model to use.")
    prompt: List[Dict[str, str]] = Field(
        default=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "Please generate some text based on the following prompt."}
        ],
        description="The User Prompt comprises of custom instructions provided by the user."
    )
    max_tokens: int = Field(default=250, description="The maximum number of tokens to generate.")
    temperature: float = Field(default=1.0, description="Sampling temperature for generation.")
    result_type: str = Field(default='assistant', description="Indicates the response type: 'raw' or 'assistant'.")
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "model_name": "gpt-3",
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Generate a response to this prompt."}
                ],
                "max_tokens": 100,
                "temperature": 1.0,
                "result_type": "assistant"
            }
        }

class MountModelRequest(BaseModel):
    model_name: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="The Hugging Face model name."
    )
    model_type: str = Field(
        default="translation",
        description=(
            "Type of model to mount. Supported model types: "
            "'translation', 'text2text-generation', 'text-generation', 'llama'."
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
        # Disable protected namespaces if not required, otherwise remove or adjust as needed
        protected_namespaces = ()      

class DownloadModelRequest(BaseModel):
    client_id: str = Field(
        ...,
        description="Unique identifier for the client"
    )
    model_name: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="The Hugging Face model name"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="The Hugging Face API key (optional). If provided, it will override the default token."
    )

    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "client_id": "unique_client_id_123",
                "model_name": "facebook/nllb-200-distilled-600M",
                # "api_key": "your_huggingface_api_key_here"  # Optional
            }
        }

class ModelRequest(BaseModel):
    model_name: str = Field(
        default="facebook/nllb-200-distilled-600M",
        description="The Hugging Face model name"
    )

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