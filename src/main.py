import os
import sys
from dotenv import load_dotenv
import logging
from fastapi import FastAPI
from helpers import setup_logging
from api import router
import uvicorn

def get_env_path():
    """
    Determines the path to the .env file based on the execution context.
    Handles both normal and frozen (e.g., PyInstaller) environments.
    """
    env_path = os.path.join(os.getcwd(), '.env')
    if os.path.exists(env_path):
        return env_path
    elif getattr(sys, 'frozen', False):
        # If the application is frozen, .env should be in the temporary directory
        env_path = os.path.join(sys._MEIPASS, '.env')
        if os.path.exists(env_path):
            return env_path
        else:
            raise FileNotFoundError(".env file not found in the temporary directory!")
    else:
        raise FileNotFoundError(".env file not found in the current directory!")

try:
    env_path = get_env_path()
    print(f"Loading .env file from: {env_path}")  # Temporary print before logging is set up
    load_dotenv(dotenv_path=env_path)
except FileNotFoundError as e:
    print(str(e))
    sys.exit(1)

setup_logging()
logger = logging.getLogger(__name__)
logger.info(f"Loaded environment variables from: {env_path}")

app = FastAPI(
    title="Hugging Face Transformers Service",
    description=(
            "This **Local LLM server** application is a Windows service application designed to provide an interface "
            "for working with Hugging Face models, specifically catering to translation and text generation tasks."
    ),
    version="1.0.9"
)
app.include_router(router)

from config import config
logger.info(f"API Name: {app.title}")
logger.info(f"API Version: {app.version}")
    
config.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "Your_Hugging_Face_API_Token")
config.HOST = os.getenv("HOST", "0.0.0.0")
config.PORT = os.getenv("PORT", "8001")
config.DOWNLOAD_DIRECTORY = os.getenv("HUGGINGFACE_MODELS_DIR", "C:/HuggingFace/Models")
    
if __name__ == "__main__":
    try:
        uvicorn.run(
            app,
            host=config.HOST,
            port=int(config.PORT),
            ws_ping_interval=600,   # 10 mins
            ws_ping_timeout=120,    # 2 mins
            timeout_keep_alive=1200  # 20 mins
        )
    except Exception as e:
        logger.exception("Failed to start the server.")
        sys.exit(1)
