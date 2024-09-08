# Hugging Face Transformers Service

## Overview

This is a FastAPI application designed to provide an intuitive and efficient interface for working with Hugging Face models, specifically catering to translation and text generation tasks. The service allows users to **download and mount** models locally, making it possible to run model inference without requiring an internet connection once the models are downloaded.

### Key Features

- **Model Management**: Users can easily download, mount, unmount, and delete Hugging Face models. Once a model is downloaded and mounted, it is accessible for inference locally, allowing for fast and efficient processing without repeated downloads.

- **Translation Functionality**: The application includes support for various translation models, enabling users to translate text between multiple languages seamlessly. Users can specify source and target languages to customize their translation tasks.

- **Text Generation**: Beyond translation, the application also supports advanced text generation tasks. Users can leverage models to generate contextual text based on provided prompts, ideal for applications such as chatbots, storytelling, or any scenario that requires dynamic text generation.

- **Interoperability**: This service enables developers to connect to the API locally from projects written in other programming languages, such as `.NET` and `Java`. This is particularly useful in environments where direct support for Transformers isn't possible, allowing developers to leverage powerful NLP capabilities without being constrained by language limitations.

- **Streaming Progress Updates**: For long-running operations such as downloading models, the application provides real-time progress updates, allowing users to monitor the status of their downloads.

<!-- - **Metrics and Monitoring**: Integrates with Prometheus and Grafana for monitoring application performance and resource utilization, ensuring that the application runs smoothly and efficiently. -->

By combining the power of Hugging Face's state-of-the-art models with the ease of FastAPI, this application empowers users to enhance their applications with robust NLP capabilities directly from their local environment.

### Supported Model Types

The following model types are supported, allowing users to leverage state-of-the-art machine learning capabilities for various natural language processing tasks:

```python
SUPPORTED_MODEL_TYPES = {
    'translation': AutoModelForSeq2SeqLM,
    'text2text-generation': AutoModelForSeq2SeqLM,
    'text-generation': AutoModelForCausalLM,
}
```

**Model Type Descriptions:**

- **Translation**: Utilizes `AutoModelForSeq2SeqLM`, enabling users to perform translations between multiple languages seamlessly.

- **Text2Text Generation**: Also uses `AutoModelForSeq2SeqLM`, designed for tasks that require transforming input text into different output text, such as summarization or question-answering.

- **Text Generation**: Leverages `AutoModelForCausalLM`, allowing users to generate coherent and contextually relevant text based on prompt inputs. Ideal for applications such as chatbots and creative writing.

<br>

## Setup Instructions

### Step 1: Check Python & Pip Installation

Ensure Python and pip are installed:

```bash
python --version
pip --version
```

If not installed, download Python from the [official website](https://www.python.org/downloads/) and install it, ensuring you select the option to add Python to your PATH.

### Step 2 Clone the Repository

```bash
git clone https://github.com/RWS/Hugging-Face-Transformers-Service.git
```

### Step 3: Create a Virtual Environment

Navigate to your project root in the terminal and create a virtual environment:

```bash
cd Hugging-Face-Transformers-Service
python -m venv venv
```

### Step 4: Activate the Virtual Environment

Activate the virtual environment:

```bash
# Windows
venv\scripts\activate

# Linux/Mac
source venv/bin/activate
```

To deactivate:

```bash
deactivate
```

### Step 5: Install Required Packages

Install the dependencies listed in `requirements.txt`:

```bash
pip install --no-cache-dir -r requirements.txt
```

### Step 6: Configure Environment Variables

To connect to Hugging Face and manage model caching, you'll need to set up your environment variables.

1. Locate the `.env` file in the root directory of the project. If it doesn't exist, you can create one.
2. Open the `.env` file and add (or update) the following lines:

```plaintext
# Directory where Hugging Face models will be cached
HUGGINGFACE_CACHE_DIR=C:/HuggingFace/model_cache

# Your Hugging Face API token for authentication
HUGGINGFACE_TOKEN=Your_Hugging_Face_API_Token

# Port to run the FastAPI application
# Adjust this to change the application's running port
PORT=8001
```

**Variable Descriptions**:

- `HUGGINGFACE_CACHE_DIR`: Specify the directory where downloaded models will be stored. Adjust the path as needed based on your system's file structure.
- `HUGGINGFACE_TOKEN`: Replace `Your_Hugging_Face_API_Token` with your actual Hugging Face API token. You can obtain this token from your Hugging Face account settings.
- `PORT`: Set the port number on which the FastAPI application will listen. The default is `8001`, but you can change this to suit your needs.

Ensure you save the changes to the `.env` file before proceeding to run the application. This configuration is essential for the application to access Hugging Face models effectively and to run the FastAPI application on the specified port.

### Step 7: Start the FastAPI Server

Run the FastAPI application:

```bash
python src/huggingface_ts.py
```

### Step 8: Access Swagger API

You can interact with the API and test its endpoints by visiting:
[http://localhost:8001/docs](http://localhost:8001/docs)

<br>

## API Endpoints

- `GET /list_models/`: Retrieve a list of all downloaded models from the `HUGGINGFACE_CACHE_DIR` directory, including their names and types.
- `POST /download_model/`: Initiate the download of a specified model from the Hugging Face Hub. Return progress updates on the download process.
- `GET /download_progress/`: Polling method to fetch the current download progress of the model, if a download is in progress.
- `POST /mount_model/`: Mount the specified model and setup the appropriate pipeline.
- `POST /unmount_model/`: Unmount the currently mounted model to free up resources.
- `DEL /delete_model/`: Delete the local files of a previously mounted model based on the model name
- `POST /translate/`: Translate input text using the mounted translation model.
- `POST /generate/`: Generate text based on the input prompt using the mounted text generation model..

### Notes

Ensure to monitor download progress through the associated API endpoints and handle errors according to the status returned.

<br>

## Docker Instructions

### Build Docker Image

To build your Docker image, run the following command from the root of your project directory (where your Dockerfile is located):

```bash
docker build --no-cache -t huggingface_ts . --progress=plain
```

### Run Docker Image

You have two options for running the Docker image based on your environment.

#### 1. Running the Docker Image on a Server

When running the Docker image on a server, set the environment variables for both the cache directory `HUGGINGFACE_CACHE_DIR` and port. Execute the following command:

```bash
docker run -d -p ${PORT}:${PORT} -e HUGGINGFACE_CACHE_DIR=/app/model_cache -e PORT=8001 huggingface_ts
```

#### 2. Running the Docker Image Locally

If you are running the Docker image locally, use the following command to mount the cache directory `C:/HuggingFace/model_cache` and specify the port from your host to the `/app/model_cache` directory inside the container. This allows the application to access and store downloaded models locally:

```bash
docker run -d -p ${PORT}:${PORT} -e HUGGINGFACE_CACHE_DIR=/app/model_cache -e PORT=8001 -v C:/HuggingFace/model_cache:/app/model_cache huggingface_ts
```

### View Running Containers

To view the list of running Docker containers, use the following command:

```bash
docker ps
```

### Stopping a Docker Container

To stop a running Docker container, you need the container ID. You can find the ID from the `docker ps` command. Use the following command to stop the container:

```bash
docker stop <container_id>
```

### Debugging the Docker Container

If you need to debug or interact with the running Docker container, you can access it using the following command (replace `<container_id>` with your actual container ID):

```bash
docker exec -it <container_id> /bin/bash
```

Once inside the container, you can navigate to the mounted directory to check the downloaded models:

```bash
ls /app/model_cache
```

To exit the interactive shell in the container, simply type:

```bash
exit
```

### Example Debugging Commands

Here is an example of how you might debug or navigate inside the container:

```bash
# Access the container's shell
docker exec -it <container_id> /bin/bash

# List the contents of the model cache directory
ls /app/model_cache

# Exit the container's shell
exit

# Stop the container
docker stop <container_id>

```
