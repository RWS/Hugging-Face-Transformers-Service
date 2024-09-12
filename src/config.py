import os

class Config:
    PORT = os.getenv("PORT", "8001").strip()
    DOWNLOAD_DIRECTORY = os.getenv("HUGGINGFACE_CACHE_DIR", "/app/model_cache").strip()
    HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN").strip()

config = Config()