from fastapi import FastAPI
from config import config
from helpers import setup_logging
from api import router

setup_logging()

app = FastAPI(
    title="Hugging Face Transformers Service",
    description="This **Local LLM server** application is a Windows service application designed to provide an intuitive and efficient interface for working with Hugging Face models, specifically catering to translation and text generation tasks.",
    version="1.0.7"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    print(f"API Name: {app.title}")
    print(f"API Version: {app.version}")
    uvicorn.run(app, host=config.HOST, 
                port=int(config.PORT),
                # log_config=None, 
                # log_level="info", 
                ws_ping_interval=600, #10 mins
                ws_ping_timeout=120, # 2 mins
                timeout_keep_alive=1200) 

