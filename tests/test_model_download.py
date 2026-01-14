import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# -------------------------------
# Make src discoverable
# Add src folder to path so that imports from src work in tests
# -------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from app import app
from api import model_states
from api import config

client = TestClient(app)

# -------------------------------
# Reset model_states before each test
# -------------------------------
def setup_function():
    model_states.clear()


# -------------------------------
# Unit Tests for /v1/model/download
# -------------------------------
@patch("api.download_model")
@patch("api.os.makedirs")
def test_download_model_normal(mock_makedirs, mock_download_model):
    """
        Test normal download initiation.

        - Sends a download request for a model.
        - Verifies HTTP 200 response and message.
        - Ensures background task is scheduled and download directory is created.
    """
    payload = {
        "client_id": "client1",
        "model_name": "test-model",
        "files_to_download": ["file1.bin"]
    }

    response = client.post("/v1/model/download", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Download started."
    assert payload["model_name"] in data["message"]

    # Background task should be scheduled
    mock_download_model.assert_called_once_with(
        client_id="client1",
        model_name="test-model",
        api_key=ANY,
        files_to_download=["file1.bin"]
    )

    # Ensure download directory is created
    mock_makedirs.assert_called_once_with(config.DOWNLOAD_DIRECTORY, exist_ok=True)


def test_download_model_active_download():
    """
        Test behavior when client already has an active download.

        - Sets is_downloading flag for client.
        - Expects HTTP 400 response indicating download already in progress.
    """
    model_states["client1"] = {"is_downloading": True}

    payload = {
        "client_id": "client1",
        "model_name": "test-model",
        "files_to_download": None
    }

    response = client.post("/v1/model/download", json=payload)
    assert response.status_code == 400
    assert "currently in progress" in response.json()["detail"]


@patch("api.download_model")
@patch("api.os.makedirs")
def test_download_model_api_key_used(mock_makedirs, mock_download_model):
    """
        Test that API key provided in headers is used for the download.

        - Sends POST request with api-key header.
        - Verifies download_model is called with the correct api_key.
    """
    payload = {
        "client_id": "client2",
        "model_name": "test-model",
        "files_to_download": None
    }

    headers = {"api-key": "dummy-key"}
    client.post("/v1/model/download", json=payload, headers=headers)

    args, kwargs = mock_download_model.call_args
    assert kwargs["api_key"] == "dummy-key"

@patch("api.download_model")
@patch("api.os.makedirs")
def test_download_model_api_key_fallback(mock_makedirs, mock_download_model):
    """
        Test fallback to config token when API key is not provided in headers.

        - Ensures download_model is called with config.HUGGINGFACE_TOKEN.
    """
    payload = {
        "client_id": "client3",
        "model_name": "test-model",
        "files_to_download": None
    }

    client.post("/v1/model/download", json=payload)

    args, kwargs = mock_download_model.call_args
    assert kwargs["api_key"] == config.HUGGINGFACE_TOKEN

@patch("api.download_model")
@patch("api.os.makedirs")
def test_download_model_empty_files_list(mock_makedirs, mock_download_model):
    """
            Test that an empty files_to_download list is handled correctly.

            - Ensures download_model receives an empty list without errors.
    """
    payload = {
        "client_id": "client4",
        "model_name": "test-model",
        "files_to_download": []
    }

    client.post("/v1/model/download", json=payload)

    args, kwargs = mock_download_model.call_args
    assert kwargs["files_to_download"] == []

def test_download_model_missing_client_id():
    """
    Test that missing client_id raises validation error (HTTP 422).

    - FastAPI will automatically validate required fields.
    """
    payload = {
        "model_name": "test-model",
        "files_to_download": ["file1.bin"]
    }

    response = client.post("/v1/model/download", json=payload)
    assert response.status_code == 422