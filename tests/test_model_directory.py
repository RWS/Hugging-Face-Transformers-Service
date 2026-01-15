import sys
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

# -------------------------------
# Mock heavy dependencies
# Mocking datasets, transformers, and torch to avoid heavy imports during testing
# -------------------------------
sys.modules['datasets'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

# -------------------------------
# Make src discoverable
# Add src folder to path so that imports from src work in tests
# -------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# -------------------------------
# Unit Tests for v1/model/directory
# -------------------------------

@patch("api.os.path.exists", return_value=True)  # Mock the directory to exist
def test_get_download_path_no_model(mock_exists):
    """
    Test v1/model/directory endpoint without providing a model name.
    Should return the default download path.
    """
    response = client.get("/v1/model/directory")
    assert response.status_code == 200
    data = response.json()
    assert "path" in data
    assert data["path"] != ""

@patch("api.os.path.exists", return_value=True)
def test_get_download_path_with_model(mock_exists):
    """Test v1/model/directory with a model name."""
    model_name = "test-model"
    response = client.get(f"/v1/model/directory?model_name={model_name}")
    assert response.status_code == 200
    data = response.json()
    assert model_name.replace("/", "--") in data["path"]


@patch("api.os.path.exists", return_value=False)  # force the directory to appear missing
def test_get_download_path_directory_missing(mock_exists):
    """
    Test v1/model/directory endpoint with a valid model name.

    Should return a path that includes the sanitized model name.
    """
    response = client.get("/v1/model/directory")

    assert response.status_code == 500
    json_data = response.json()
    assert "Base download directory does not exist" in json_data["detail"]

@patch("api.os.path.exists", return_value=True)
def test_get_download_path_with_slash_in_model(mock_exists):
    """
    Test v1/model/directory endpoint with a model name that contains a slash.
    Should sanitize slashes in model_name.
    """
    model_name = "test/model"
    response = client.get(f"/v1/model/directory?model_name={model_name}")
    assert response.status_code == 200
    data = response.json()
    assert "test--model" in data["path"]