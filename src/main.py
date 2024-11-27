from fastapi import FastAPI
from config import config
from api import router

app = FastAPI(
    title="Hugging Face Transformers Service",
    description="This is a FastAPI application designed to provide an intuitive and efficient interface for working with Hugging Face models, specifically catering to translation and text generation tasks. The service allows users to **download and mount** models locally, making it possible to run model inference without requiring an internet connection once the models are downloaded.",
    version="1.0.3"
)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    print(f"API Name: {app.title}")
    print(f"API Version: {app.version}")
    uvicorn.run(app, host=config.HOST, port=int(config.PORT))