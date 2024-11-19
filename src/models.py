from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union


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
    model_name: str = Field(default="facebook/nllb-200-distilled-600M", description="The Hugging Face model name")
    model_type: str = Field(default="translation", description="Type of model to mount. Supported model types: 'translation', 'text2text-generation', 'text-generation', 'llama'.")
    source_language: Optional[str] = Field(default="eng_Latn", description="[Optional] Language code for the source language (e.g., 'eng_Latn' for English).")
    target_language: Optional[str] = Field(default="ita_Latn", description="[Optional] Language code for the target language (e.g., 'ita_Latn' for Italian).")

    class Config:
        protected_namespaces = ()  
        json_schema_extra  = {
            "example": {
                "model_name": "facebook/nllb-200-distilled-600M",
                "model_type": "translation",
                "source_language": "eng_Latn",
                "target_language": "ita_Latn"
            }
        }

class GeneratedResponse(BaseModel):
    generated_response: Union[str, List[str]] = Field(
        description="The generated text from the model based on the provided prompt, either as a string or a list of strings.",
        example="Il gatto è sul tavolo."
    )

    class Config:
        protected_namespaces = ()  # Disable protected namespaces          

class ModelRequest(BaseModel):
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
                "model_name": "facebook/nllb-200-distilled-600M",
                # "api_key": "your_huggingface_api_key_here"  # Optional
            }
        }

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    model_mounted: bool  # Boolean type
    model_size_bytes: str

    class Config:
        protected_namespaces = ()  

class LocalModel:
    def __init__(self, model_name: str, model, model_type: str, tokenizer, pipeline):
        self.model_name = model_name
        self.model = model
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.pipeline = pipeline

    class Config:
        protected_namespaces = () 