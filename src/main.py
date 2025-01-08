import os
import sys
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
from config import config
import uvicorn
from app import app
import multiprocessing 

def get_base_directory() -> str:
    """
    Returns the base directory where the executable is located.
    Handles both frozen (PyInstaller) and unfrozen states.
    """
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle, the PyInstaller bootloader
        # extends the sys module by a flag frozen=True and sets the app
        # path into variable _MEIPASS'.
        return os.path.dirname(sys.executable)
    else:
        # If unfrozen, return the directory of the script file.
        return os.path.dirname(os.path.abspath(__file__))
    
def setup_logging():
    base_dir = get_base_directory()
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)  

    log_file = os.path.join(log_dir, "hfts.log")
   
    rotating_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=20, # Keep up to 20 backup files
        encoding="utf-8",
        delay=False
    )

    # Create a logging format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    rotating_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    logger.addHandler(rotating_handler)
  
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Base Directory: {base_dir}")
    logger.info(f"Log Directory: {log_dir}")
    logger.info(f"Log File: {log_file}")

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

if __name__ == "__main__":
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
    
    # Set configuration from environment variables
    config.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "Your_Hugging_Face_API_Token")
    config.HOST = os.getenv("HOST", "0.0.0.0")
    config.PORT = os.getenv("PORT", "8001")
    config.DOWNLOAD_DIRECTORY = os.getenv("HUGGINGFACE_MODELS_DIR", "C:/HuggingFace/Models")
        
    logger.info(f"API Name: {app.title}")
    logger.info(f"API Version: {app.version}")
        
    try:
        multiprocessing.freeze_support() 
        uvicorn.run(
            app,
            host=config.HOST,
            port=int(config.PORT),
            ws_ping_interval=600,   # 10 mins
            ws_ping_timeout=120,    # 2 mins
            timeout_keep_alive=1200  # 20 mins / TODO review this so that we don't keep zombie connections open for such a long time...
        )
    except Exception as e:
        logger.exception("Failed to start the server.")
        sys.exit(1)