import os

class Config:
    """
    Configuration class to retrieve environment variables.
    Assumes that the .env file has already been loaded in the environment.
    """
    def __init__(self):
        self.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "Your_Hugging_Face_API_Token")
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = os.getenv("PORT", "8001")
        self.DOWNLOAD_DIRECTORY = os.getenv("HUGGINGFACE_MODELS_DIR", "C:/HuggingFace/Models")

config = Config()