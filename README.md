# Hugging Face Transformers Service

## Overview

This **Local LLM server** is a Windows service application designed to provide an interface for working with Hugging Face models, specifically catering to translation and text generation tasks.

The application is built leveraging FastAPI and Python, and it supports a wide selection of Hugging Face LLMs. It can run either as a local Windows service or be included in a Docker container, enabling users to download and mount models locally.

### Key Features

- **Model Management**: Users can easily download, mount, unmount, and delete Hugging Face models. Once a model is downloaded and mounted, it is accessible for inference locally, allowing for fast and efficient processing without repeated downloads.
- **Translation Functionality**: The application includes support for various translation models, enabling users to translate text between multiple languages.
- **Fine-Tuning**: Facilitate the fine-tuning of pretrained translation models with custom datasets, allowing users to adapt models to specific domains or language nuances.
- **Text Generation**: Leverage models to generate contextual text based on provided prompts, ideal for translation tasks that require dynamic text generation.
- **Interoperability**: This service enables developers to connect to the API locally from projects written in other programming languages, such as `.NET` and `Java`. This is particularly useful in environments where direct support for Transformers isn't possible, allowing developers to leverage powerful NLP capabilities without being constrained by language limitations.
- **Streaming Progress Updates**: For long-running operations such as downloading models, the application provides real-time progress updates, allowing users to monitor the status of their downloads.

### Supported Models

Below is an overview of the supported models categorized by their primary functionalities. This will help you choose the right model for your specific needs.

#### Translation Models

| **Model Name**                               | **Primary Uses**                                                                                                               | **Supported Languages**                                                                                                                                                                                                                                                              | **Notes**                                                                                                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **facebook/nllb-200-distilled-600M**         | Machine translation research, especially for low-resource languages. Supports single-sentence translation among 200 languages. | eng_Latn, ita_Latn, deu_Latn, fra_Latn, spa_Latn, ell_Grek, jpn_Jpan, zho_Hans, zho_Hant, etc.… [Full list in [README](https://huggingface.co/facebook/nllb-200-distilled-600M/blob/main/README.md)]                                                                                 | Trained with input lengths not exceeding 512 tokens; translating longer sequences may degrade quality. Translations cannot be used as certified translations. |
| **facebook/mbart-large-50-many-to-many-mmt** | Multilingual machine translation. Enables direct translation between any pair of 50 languages.                                 | Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), etc.… [Full list in [README](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt)] | The model can translate directly between any pair of 50 languages.                                                                                            |
| **Helsinki-NLP/opus-mt-\***                  | Various language pair translations. Offers a wide range of language directions.                                                | Varies per model (e.g., en-it, it-en, en-fr, fr-en, en-de, de-en, etc.)                                                                                                                                                                                                              | Helsinki-NLP is focused on wide language coverage, open datasets, and public pre-trained models.                                                              |

#### Text Generation Models

| **Model Name**                                       | **Primary Uses**                                                                                      | **Supported Languages**          | **Notes**                                                                                                                                                                                                                      |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **lmstudio-community/Llama-3.2-3B-Instruct-GGUF**    | Instruction-tuned large language model for versatile text generation tasks.                           | Multilingual (primarily English) | Part of the Meta Llama collection, available in 3B, 8B, 70B, and 405B sizes. English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported.                                                  |
| **lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF** | Enhanced text generation with instruction tuning for better contextual understanding.                 | Multilingual (primarily English) | Part of the Meta Llama collection. English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported.                                                                                            |
| **meta-llama/Llama-3.1-8B-Instruct**                 | Instruction-based text generation with improved performance over previous versions.                   | Multilingual (primarily English) | Part of the Meta Llama collection. English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported.                                                                                            |
| **meta-llama/Llama-3.2-3B-Instruct**                 | Advanced conversational tasks and assistant-like interactions.                                        | Multilingual (primarily English) | Part of the Meta Llama collection. English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported.                                                                                            |
| **bartowski/Phi-3.5-mini-instruct-GGUF**             | Lightweight, high-quality text generation with reasoning capabilities. Supports long context lengths. | Multilingual (primarily English) | Supported languages: Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Hebrew, Hungarian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian |
| **microsoft/Phi-3.5-mini-instruct**                  | Similar to Phi-3.5-mini-instruct-GGUF with Microsoft optimizations.                                   | Multilingual (primarily English) | Supported languages: Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Hebrew, Hungarian, Italian, Japanese, Korean, Norwegian, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian |
| **bartowski/Mistral-Nemo-Instruct-2407-GGUF**        | Instruction fine-tuned large language model with enhanced performance.                                | Multilingual (primarily English) | Jointly trained by Mistral AI and NVIDIA.                                                                                                                                                                                      |
| **mistralai/Mistral-Nemo-Instruct-2407**             | Instruction fine-tuned version of Mistral-Nemo-Base-2407.                                             | Multilingual (primarily English) | Significantly outperforms existing models of similar size.                                                                                                                                                                     |
| **google/gemma-2-9b**                                | General-purpose text generation, question answering, summarization, and reasoning.                    | English                          | Lightweight and deployable on limited-resource environments. Open weights available for pre-trained and instruction-tuned variants.                                                                                            |
| **bartowski/Mistral-7B-Instruct-v0.3-GGUF**          | Instruction fine-tuned model for detailed and context-aware text generation.                          | English                          |                                                                                                                                                                                                                                |
| **Mistral-7B-Instruct-v0.3**                         | Enhanced instruction-tuned text generation model.                                                     | English                          |                                                                                                                                                                                                                                |

### Fine-Tuning Translation Models

Enable users to **fine-tune pretrained translation models** with their own datasets, enhancing the model's performance tailored to specific domains or language nuances.

#### How It Works

1. **Provide a Fine-Tuning Request**: Submit a POST request to the `/fine-tune/` endpoint with necessary parameters, including the path to the pretrained model, dataset, and desired training configurations.
2. **Real-Time Progress Updates**: Monitor the fine-tuning progress in real-time via a WebSocket connection established at `/ws/progress/{client_id}`.
3. **Receive the Fine-Tuned Model**: Upon completion, the fine-tuned model is saved to the specified output directory, ready for deployment or further use.

#### Example Usage

- **Initiate Fine-Tuning**:

```json
{
  "client_id": "unique_client_id_123",
  "model_path": "/path/to/pretrained/model",
  "output_dir": "/path/to/save/fine-tuned/model",
  "validation_file": "/path/to/validation_data.csv",
  "data_file": "/path/to/data.csv",
  "source_lang": "en_XX",
  "target_lang": "it_IT",
  "num_train_epochs": 4,
  "per_device_train_batch_size": 2,
  "per_device_eval_batch_size": 2,
  "learning_rate": 3e-5,
  "weight_decay": 0.01,
  "max_length": 512,
  "save_steps": 10,
  "save_total_limit": 2
}
```

- **Establish WebSocket Connection**:
    Connect to `/ws/progress/unique_client_id_123` to receive real-time updates during the fine-tuning process.

### Supported Languages

**Translation Models** support a variety of languages. Refer to the detailed list in the table above under each model or access the full list in the [Helsinki-NLP documentation](https://github.com/Helsinki-NLP).

<br>

## Setup Instructions

### Step 1: Check Python & Pip Installation

Ensure Python and pip are installed:

```bash
python --version
pip --version
```

If not installed, download Python from the [official website](https://www.python.org/downloads/) and install it, ensuring you select the option to add Python to your PATH.

### Step 2: Clone the Repository

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

### Step 5: Install Microsoft Visual C++ Redistributables

These redistributables provide essential runtime components required by the application. You can download the latest supported versions directly from Microsoft's official [latest supported Visual C++ downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

### Step 6: Install Required Packages

Install the dependencies listed in `requirements.txt`:

```bash
pip install --no-cache-dir -r requirements.txt
```

### Step 7: Configure Environment Variables

To connect to Hugging Face and manage model caching, you'll need to set up your environment variables.

1. Locate the `.env` file in the root directory of the project. If it doesn't exist, you can create one.
2. Open the `.env` file and add (or update) the following lines:

```plaintext
# Directory where Hugging Face models will be cached
HUGGINGFACE_MODELS_DIR=C:/HuggingFace/model_cache

# Your Hugging Face API token for authentication
HUGGINGFACE_TOKEN=Your_Hugging_Face_API_Token

# HOST IP address to run the REST API application
HOST=0.0.0.0

# Port number to run the REST API application
PORT=8001
```

**Variable Descriptions**:

- `HUGGINGFACE_MODELS_DIR`: Specify the directory where downloaded models will be stored. Adjust the path as needed based on your system's file structure.
- `HUGGINGFACE_TOKEN`: Replace `Your_Hugging_Face_API_Token` with your actual Hugging Face API token. You can obtain this token from your Hugging Face account settings.
- `HOST`: Set the host IP address for the Local LLM server. The default is `0.0.0.0` to allow access from any IP.
- `PORT`: Set the port number on which the Local LLM server will listen. The default is `8001`, but you can change this to suit your needs.

Ensure you save the changes to the `.env` file before proceeding to run the application. This configuration is essential for the application to access Hugging Face models effectively and to run the Local LLM server on the specified port.

### Step 8: Start the Local LLM Server

Run the Local LLM server:

```bash
python src/main.py
```

### Step 9: Access Swagger API & Documentation

- **Swagger API**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **API Documentation**: [https://developers.rws.com/HFTS-api-docs/redoc.html](https://developers.rws.com/HFTS-api-docs/redoc.html)
<!-- [http://localhost:8001/redoc](http://localhost:8001/redoc) -->

<br>

## Installer

### Step 1: Compile the Application with PyInstaller

To create an executable for your Local LLM server, use PyInstaller as follows:

```bash
pyinstaller HuggingFace-TS.spec
```

### Step 2: Copy the `.env` File

After compiling your application, ensure that the `.env` file is present in the same directory as the executable (`HuggingFace-TS.exe`). Users can also create or modify this file based on their configuration needs.

- **Editing the `.env` File**: The `.env` file can be opened and modified using any text editor (e.g., Notepad, Visual Studio Code).

### Step 3: Start the Local LLM Server

Run the Local LLM server by executing `HuggingFace-TS.exe` from the `dist` directory:

```bash
cd dist
HuggingFace-TS.exe
```

<br>

## API Endpoints

- `GET /v1/models`: Retrieve a list of all downloaded models from the `HUGGINGFACE_MODELS_DIR` directory, including their names and types.
- `GET /v1/model/info`: Retrieve model information, including configuration details and types supported.
- `GET /v1/model/directory`: Retrieves the current download directory, including the `model_name` if provided and not empty.
- `GET /v1/model/files`: Retrieves the list of available files in the specified Hugging Face model repository, including each file's size when available.
- `POST /v1/model/download`: Initiate the download of a specified model from the Hugging Face Hub. Return progress updates on the download process.
- `DEL /v1/model/delete`: Delete the local files of a previously mounted model based on the model name.
- `POST /v1/model/mount/`: Mount the specified model and set up the appropriate pipeline.
- `POST /v1/model/unmount`: Unmount the currently mounted model to free up resources.
- `POST /v1/model/fine-tune`: Initiate the fine-tuning of a specified translation model with custom parameters and data. Receive real-time progress updates via WebSocket.
- `POST /v1/translate`: Translate input text using the mounted translation model.
- `POST /v1/completions`: Generate text based on the input prompt using the specified text generation model.
- `POST /v1/chat/completions`: Generate chat-style completion using the specified text generation model.
- `WS /v1/ws/progress/{client_id}`: Establish a WebSocket connection to receive real-time progress updates for model download operations.

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

When running the Docker image on a server, set the environment variables for both the cache directory `HUGGINGFACE_MODELS_DIR` and `PORT`. Execute the following command:

```bash
docker run -d -p ${PORT}:${PORT} -e HUGGINGFACE_MODELS_DIR=/app/model_cache -e PORT=8001 huggingface_ts
```

#### 2. Running the Docker Image Locally

If you are running the Docker image locally, use the following command to mount the cache directory `C:/HuggingFace/Models` and specify the port from your host to the `/app/model_cache` directory inside the container. This allows the application to access and store downloaded models locally:

```bash
docker run -d -p ${PORT}:${PORT} -e HUGGINGFACE_MODELS_DIR=/app/model_cache -e PORT=8001 -v C:/HuggingFace/Models:/app/model_cache huggingface_ts
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

---

## Additional Notes

- **Model Updates**: Regularly check for updates to the models you are using to ensure optimal performance and access to the latest features.
- **Resource Management**: Monitor system resources when running multiple models or handling large datasets to prevent performance degradation.
