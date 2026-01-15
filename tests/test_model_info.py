import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import os

# -------------------------------
# Make src discoverable
# Add src folder to path so that imports from src work in tests
# -------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

MODEL_NAME = "test-model"

# -------------------------------
# Unit Tests for /v1/model/info
# -------------------------------

@patch("api.os.path.exists")
@patch("api.AutoConfig.from_pretrained")
def test_get_model_info_config_json(mock_from_pretrained, mock_exists):
    """
    Test /v1/model/info returns model configuration if config.json exists.

    - Mocks os.path.exists to simulate presence of config.json.
    - Mocks AutoConfig.from_pretrained().to_dict() to return dummy config.
    - Expects HTTP 200 with 'config' in response.
    """
    mock_exists.return_value = True
    mock_from_pretrained.return_value.to_dict = MagicMock(return_value={"layers": 12})

    response = client.get(f"/v1/model/info?model_name={MODEL_NAME}&return_type=config")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == MODEL_NAME
    assert "config" in data
    assert data["config"]["layers"] == 12


@patch("api.os.path.exists")
@patch("api.glob.glob")
def test_get_model_info_gguf_files(mock_glob, mock_exists):
    """
    Test /v1/model/info returns minimal info if no config.json but *.gguf files exist.

    - Mocks os.path.exists to simulate missing config.json.
    - Mocks glob.glob to return one .gguf file.
    - Expects HTTP 200 with 'minimal_info' including file names and default model type.
    """
    mock_exists.return_value = False
    mock_glob.return_value = ["/fake/path/model.gguf"]

    response = client.get(f"/v1/model/info?model_name={MODEL_NAME}&return_type=config")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == MODEL_NAME
    assert "minimal_info" in data
    assert data["minimal_info"]["model_type"] == "text-generation"
    assert data["minimal_info"]["file_names"] == ["model.gguf"]


@patch("api.os.path.exists")
@patch("api.glob.glob")
def test_get_model_info_no_config_no_gguf(mock_glob, mock_exists):
    """
    Test /v1/model/info returns minimal info when neither config.json nor .gguf files exist.

    - Mocks os.path.exists and glob.glob to simulate missing files.
    - Expects HTTP 200 with 'minimal_info', file_names empty, and model_type 'unknown'.
    """
    mock_exists.return_value = False
    mock_glob.return_value = []

    response = client.get(f"/v1/model/info?model_name={MODEL_NAME}&return_type=config")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == MODEL_NAME
    assert "minimal_info" in data
    assert data["minimal_info"]["file_names"] == []
    assert data["minimal_info"]["model_type"] == "unknown"


@patch("api.fetch_model_info")
def test_get_model_info_api(mock_fetch):
    """
    Test /v1/model/info with return_type=info fetches data from Hugging Face API.

    - Mocks fetch_model_info to return a dummy model object.
    - Expects HTTP 200 with 'info' containing model_id.
    """
    mock_model_info = MagicMock()
    mock_model_info.modelId = MODEL_NAME
    mock_model_info.pipeline_tag = "translation"
    mock_model_info.transformers_info = {"dummy": True}
    mock_model_info.card_data = {}
    mock_model_info.siblings = []
    mock_model_info.library_name = "transformers"
    mock_model_info.widget_data = {}
    mock_model_info.config = {"layers": 12}
    mock_model_info.spaces = []
    mock_model_info.tags = ["tag1"]
    mock_model_info.downloads = 100
    mock_model_info.lastModified = "2025-12-08"
    mock_model_info.safetensors = True

    mock_fetch.return_value = mock_model_info

    response = client.get(f"/v1/model/info?model_name={MODEL_NAME}&return_type=info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == MODEL_NAME
    assert "info" in data
    assert data["info"]["model_id"] == MODEL_NAME


def test_get_model_info_invalid_return_type():
    """
    Test /v1/model/info returns HTTP 400 for invalid return_type.

    - Expects an error message indicating invalid return_type.
    """
    response = client.get(f"/v1/model/info?model_name={MODEL_NAME}&return_type=invalid")
    assert response.status_code == 400
    data = response.json()
    assert "Invalid return_type" in data["detail"]


@patch("api.fetch_model_info", side_effect=Exception("API failure"))
def test_get_model_info_api_error(mock_fetch):
    """
    Test /v1/model/info handles exceptions from Hugging Face API.

    - If fetch_model_info raises an exception, API should return HTTP 500 with the exception message.
    """
    response = client.get(f"/v1/model/info?model_name={MODEL_NAME}&return_type=info")
    assert response.status_code == 500
    assert "API failure" in response.json()["detail"]