from fastapi import FastAPI
from dotenv import load_dotenv
from api import router
import os

load_dotenv()

app = FastAPI(
    title="Hugging Face Transformers Service",
    description="This is a FastAPI application designed to provide an intuitive and efficient interface for working with Hugging Face models, specifically catering to translation and text generation tasks. The service allows users to **download and mount** models locally, making it possible to run model inference without requiring an internet connection once the models are downloaded.",
    version="1.0.0"
)


app.include_router(router)

port = os.getenv("PORT", "8001").strip()  # Ensure extra whitespace is removed

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(port))