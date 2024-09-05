# Hugging Face Transformers Service

## Overview

This is a FastAPI application that enables users to download and mount Hugging Face models for translation tasks.

## Setup Instructions

### Step 1: Check Python & Pip Installation

Ensure Python and pip are installed:

```bash
python --version
pip --version
```

If not installed, download Python from the [official website](https://www.python.org/downloads/) and install it, ensuring you select the option to add Python to your PATH.

### Step 2: Create a Virtual Environment

Navigate to your project root in the terminal and create a virtual environment:

```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment

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

### Step 4: Install Required Packages

Install the dependencies listed in `requirements.txt`:

```bash
RUN pip install --no-cache-dir -r requirements.txt
```

### Step 5: Configure Environment Variables

Open the `.env` file located at the project's root directory and set the cache directory for Hugging Face models, if necessary:

```plaintext
HUGGINGFACE_CACHE_DIR=C:/HuggingFace/model_cache
```

### Step 6: Start the FastAPI Server

Run the FastAPI application:

```bash
python src/huggingface_ts.py
```

### Step 7: Access Swagger API

You can interact with the API and test its endpoints by visiting:
[http://localhost:8001/docs](http://localhost:8001/docs)

## API Endpoints

- `POST /download_model/`: Initiate the download of a specified model from the Hugging Face Hub. Return progress updates on the download process.
- `GET /download_progress/`: Polling method to fetch the current download progress of the model, if a download is in progress.
- `POST /mount_model/`: Mount the specified model and setup the appropriate pipeline.
- `POST /unmount_model/`: Unmount the currently mounted model to free up resources.
- `POST /delete_model/`: Delete the local files of a previously mounted model based on the model name
- `POST /translate/`: Translate input text using the mounted translation model.
- `POST /generate/`: Generate text based on the input prompt using the mounted text generation model..

## Notes

Ensure to monitor download progress through the associated API endpoints and handle errors according to the status returned.

## Docker Instructions

### Build Docker Image

To build your Docker image, run the following command from the root of your project directory (where your Dockerfile is located):

```bash
docker build --no-cache -t huggingface_ts . --progress=plain
```

### Run Docker Image

You have two options for running the Docker image based on your environment.

#### 1. Running the Docker Image on a Server

When running the Docker image on a server, set the environment variable `HUGGINGFACE_CACHE_DIR` to specify the path in the Docker container. Execute the following command:

```bash
docker run -d -p 8001:8001 -e HUGGINGFACE_CACHE_DIR=/app/model_cache huggingface_ts
```

#### 2. Running the Docker Image Locally

If you are running the Docker image locally, use the following command to mount the `C:/HuggingFace/model_cache` directory from your host to the `/app/model_cache` directory inside the container. This allows the application to access and store downloaded models locally:

```bash
docker run -d -p 8001:8001 -e HUGGINGFACE_CACHE_DIR=/app/model_cache -v C:/HuggingFace/model_cache:/app/model_cache huggingface_ts
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
