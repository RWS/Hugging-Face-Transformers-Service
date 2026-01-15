import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# -------------------------------
# Make src discoverable
# Add src folder to path so that imports from src work in tests
# -------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app import app  # app.py includes the router

client = TestClient(app)

# -------------------------------
# Helper mocks
# -------------------------------

class MockMountedModel:
    """
        Fake mounted model object for testing.

        Attributes:
            model_name: Name of the model.
            properties: Optional dictionary of model properties.
            file_name: Name of the currently loaded file.
    """
    def __init__(self, name, properties=None, file_name=None):
        self.model_name = name
        self.properties = properties or {}
        self.file_name = file_name

# -------------------------------
# Unit Tests for v1/models
# -------------------------------

@patch("api.os.listdir")
@patch("api.os.path.isdir", return_value=True)
@patch("api.infer_model_type")
@patch("api.get_directory_size")
@patch("api.format_size")
@patch("api.glob.glob")
@patch("api.model_state")
def test_models_endpoint_normal(mock_model_state, mock_glob, mock_format_size, mock_get_size,
                                mock_infer_type, mock_isdir, mock_listdir):
    """
    Test normal case with both mounted and unmounted models.

    - Mocks os.listdir to simulate model folders including hidden and lock files.
    - Mocks mounted models in model_state.
    - Mocks glob.glob to return GGUF files for mounted models.
    - Mocks size formatting and directory size calculation.
    - Asserts hidden and lock folders are skipped.
    - Checks that mounted models include properties, loaded_file_name, file_names, and size.
    - Checks unmounted models have default values for missing attributes.
    """
    mock_listdir.return_value = ["mounted-model", "unmounted-model", ".hidden", ".locks"]
    mock_infer_type.side_effect = ["text-generation", "translation", "text-generation", "text-generation"]
    mock_get_size.side_effect = [123456789, 98765432, 55555, 0]
    mock_format_size.side_effect = ["117.74 MB", "94.21 MB", "54.25 KB", "0 B"]
    mock_glob.side_effect = [
        ["file1.gguf", "file2.gguf"],  # mounted text-gen model
        [],                            # unmounted translation model
        [],                            # hidden
        []                             # locks
    ]

    mock_model_state.models = [
        MockMountedModel(name="mounted-model", properties={"src_lang": "eng_Latn"}, file_name="file1.gguf")
    ]

    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2

    mounted = next(m for m in data if m["model_name"] == "mounted-model")
    unmounted = next(m for m in data if m["model_name"] == "unmounted-model")

    assert mounted["model_mounted"] is True
    assert mounted["properties"] == {"src_lang": "eng_Latn"}
    assert mounted["loaded_file_name"] == "file1.gguf"
    assert mounted["file_names"] == ["file1.gguf", "file2.gguf"]
    assert mounted["model_size_bytes"] == "117.74 MB"

    assert unmounted["model_mounted"] is False
    assert unmounted["properties"] == {}
    assert unmounted["loaded_file_name"] is None
    assert unmounted["file_names"] is None
    assert unmounted["model_size_bytes"] == "94.21 MB"

@patch("api.os.listdir", return_value=[])
@patch("api.os.path.isdir", return_value=True)
def test_models_endpoint_empty(mock_isdir, mock_listdir):
    """
    Test /v1/models when DOWNLOAD_DIRECTORY exists but has no model folders.

    - Mocks os.listdir to return an empty list.
    - Expects HTTP 200 with an empty list as response.
    """
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data == []

@patch("api.os.listdir", side_effect=Exception("Cannot access directory"))
def test_models_endpoint_error(mock_listdir):
    """
    Test /v1/models when accessing the download directory raises an exception.

    - Expects HTTP 500 with an appropriate error message.
    """
    response = client.get("/v1/models")
    assert response.status_code == 500
    data = response.json()
    assert "Error accessing model cache" in data["detail"]

@patch("api.os.listdir")
@patch("api.os.path.isdir", return_value=True)
@patch("api.infer_model_type", return_value="text-generation")
@patch("api.get_directory_size", return_value=1024)
@patch("api.format_size", return_value="1.00 KB")
@patch("api.glob.glob", return_value=[])
@patch("api.model_state")
def test_models_endpoint_textgen_no_gguf(mock_model_state, mock_glob, mock_format_size, mock_get_size,
                                         mock_infer_type, mock_isdir, mock_listdir):
    """
    Test a text-generation model with no GGUF files.

    - Ensures file_names is None when no GGUF files are present.
    """
    mock_listdir.return_value = ["text-gen-model"]
    mock_model_state.models = []
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["file_names"] is None

@patch("api.os.listdir")
@patch("api.os.path.isdir", return_value=True)
@patch("api.infer_model_type", return_value="translation")
@patch("api.get_directory_size", return_value=1024)
@patch("api.format_size", return_value="1.00 KB")
@patch("api.glob.glob", return_value=[])
@patch("api.model_state")
def test_models_endpoint_unmounted_model(mock_model_state, mock_glob, mock_format_size, mock_get_size,
                                         mock_infer_type, mock_isdir, mock_listdir):
    """
    Test an unmounted model.

    - Should have model_mounted=False, properties={}, and loaded_file_name=None.
    """
    mock_listdir.return_value = ["unmounted-model"]
    mock_model_state.models = []
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data[0]["model_mounted"] is False
    assert data[0]["properties"] == {}
    assert data[0]["loaded_file_name"] is None