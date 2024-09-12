import os
import sys
from dotenv import load_dotenv

class Config:
    def __init__(self):
        env_path = os.path.join(os.getcwd(), '.env')        
       
        if os.path.exists(env_path):
            print("Loading .env file from: " + env_path)
        elif getattr(sys, 'frozen', False):            
            env_path = os.path.join(sys._MEIPASS, '.env')
            if not os.path.exists(env_path):
                raise FileNotFoundError(".env file not found in the temporary directory!")
            print("Loading .env file from the temporary directory: " + env_path)
        else:            
            raise FileNotFoundError(".env file not found in the current directory!")
        
        load_dotenv(dotenv_path=env_path)

        # Retrieve configuration values
        self.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or "Your_Hugging_Face_API_Token"
        self.PORT = os.getenv("PORT", "8001")
        self.DOWNLOAD_DIRECTORY = os.getenv("HUGGINGFACE_CACHE_DIR") or "C:/HuggingFace/model_cache"        

config = Config()  