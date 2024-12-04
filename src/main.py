from fastapi import FastAPI
from config import config
from api import router

app = FastAPI(
    title="Hugging Face Transformers Service",
    description="This **Local LLM server** application is a Windows service application designed to provide an intuitive and efficient interface for working with Hugging Face models, specifically catering to translation and text generation tasks.",
    version="1.0.5"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    print(f"API Name: {app.title}")
    print(f"API Version: {app.version}")
    uvicorn.run(app, host=config.HOST, port=int(config.PORT), ws_ping_interval=60)