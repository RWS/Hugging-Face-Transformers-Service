from fastapi import FastAPI
from api import router 

app = FastAPI(
    title="Hugging Face Transformers Service",
    description=(
        "This **Local LLM server** application is a Windows service application designed to provide an interface "
        "for working with Hugging Face models, specifically catering to translation and text generation tasks."
    ),
    version="1.0.10"
)

app.include_router(router)