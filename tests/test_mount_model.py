import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# -------------------------------
# Mock heavy dependencies
# Mocking datasets, transformers, and torch to avoid heavy imports during testing
# -------------------------------
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['llama_cpp'] = MagicMock()

# -------------------------------
# Make src discoverable
# Add src folder to path so that imports from src work in tests
# -------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from app import app
from api import model_state

client = TestClient(app)

# -------------------------------
# Helper functions
# -------------------------------

def mock_exists(path):
    """Custom side effect for os.path.exists during tests."""
    if path.endswith("textgen-gguf"):
        return True
    # file exists
    if path.endswith("model.gguf"):
        return True
    return False


def reset_model_state():
    """Reset model_state before each test to ensure clean environment."""
    model_state.models = []
    model_state.device = "cpu"
    model_state.cpu_cores = 2

# -------------------------------
# Unit Tests for /v1/model/mount
# -------------------------------

@patch("api.os.path.exists", return_value=True)  # Ensure directory exists
@patch("api.infer_model_type", return_value="translation")
@patch("api.get_model_type")
@patch("api.AutoTokenizer.from_pretrained")
@patch("api.pipeline")
def test_mount_translation_model(mock_pipeline, mock_tokenizer, mock_get_model_type, mock_infer, mock_exists):
    """
    Mount a translation model with properties successfully.

    - Ensures model is added to model_state.models.
    - Checks returned properties match input.
    - Verifies model_type is correctly set to "translation".
    """
    reset_model_state()
    payload = {"model_name": "translation-model", "properties": {"src_lang": "eng", "tgt_lang": "fra"}}
    response = client.post("/v1/model/mount", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "mounted successfully" in data["message"]
    assert data["properties"] == payload["properties"]
    assert len(model_state.models) == 1
    assert model_state.models[0].model_type == "translation"

@patch("api.os.path.exists", side_effect=mock_exists)
@patch("api.infer_model_type", return_value="text-generation")
@patch("api.Llama.from_pretrained")
def test_mount_text_generation_no_file(mock_llama, mock_infer, mock_exists):
    """
        Mount a text-generation model when no specific GGUF file is provided.

        - Checks model is mounted successfully using only the directory.
    """
    reset_model_state()

    # Mock os.path.exists to return True for the model directory
    def exists_side_effect(path):
        if "textgen-model" in path and not path.endswith(".gguf"):
            return True  # directory exists
        return False

    mock_exists.side_effect = exists_side_effect

    payload = {"model_name": "textgen-model"}
    response = client.post("/v1/model/mount", json=payload)
    assert response.status_code == 200
    assert "mounted successfully" in response.json()["message"]

@patch("api.os.path.exists", return_value=False)
def test_mount_model_path_missing(mock_exists):
    """Fail to mount if the model path does not exist."""
    reset_model_state()
    payload = {"model_name": "missing-model"}
    response = client.post("/v1/model/mount", json=payload)

    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]


@patch("api.os.path.exists")
@patch("api.infer_model_type", return_value="text-generation")
def test_mount_text_generation_file_invalid(mock_infer, mock_exists):
    """
        Fail mounting when the specified GGUF file does not exist.

        - Checks that HTTP 404 is returned with appropriate error detail.
    """
    from api import model_state
    model_state.models = []

    def exists_side_effect(path):
        if "textgen-model" in path and not path.endswith("invalid.gguf"):
            return True  # directory exists
        return False  # file does not exist
    mock_exists.side_effect = exists_side_effect

    payload = {"model_name": "textgen-model", "file_name": "invalid.gguf"}
    response = client.post("/v1/model/mount", json=payload)

    assert response.status_code == 404
    assert "does not exist" in response.json()["detail"]



@patch("api.infer_model_type", return_value="unsupported-type")
@patch("api.os.path.exists", return_value=True)
def test_mount_unsupported_model_type(mock_exists, mock_infer):
    """
    Fail mounting unsupported model types.

    - Expects HTTP 400 with message about unsupported type.
    """
    reset_model_state()
    payload = {"model_name": "unsupported-model"}
    response = client.post("/v1/model/mount", json=payload)

    assert response.status_code == 400
    assert "Unsupported model type" in response.json()["detail"]

def test_mount_already_mounted():
    """
    Return a message if the model is already mounted.

    - Ensures idempotent behavior: does not re-mount the same model.
    """
    reset_model_state()
    model_state.models.append(MagicMock(model_name="mounted-model"))

    payload = {"model_name": "mounted-model"}
    response = client.post("/v1/model/mount", json=payload)

    assert response.status_code == 200
    assert "already mounted" in response.json()["message"]

@patch("api.os.path.exists", side_effect=mock_exists)
@patch("api.infer_model_type", return_value="text-generation")
@patch("api.Llama.from_pretrained")
def test_mount_text_generation_with_valid_gguf(mock_llama, mock_infer, mock_exists):
    """
        Mount a text-generation model with a valid GGUF file.

        - Ensures model is added to model_state.models.
        - Checks file_name is correctly assigned.
        - Verifies success HTTP 200 response.
    """
    reset_model_state()
    payload = {"model_name": "textgen-gguf", "file_name": "model.gguf"}
    response = client.post("/v1/model/mount", json=payload)

    assert response.status_code == 200
    assert "mounted successfully" in response.json()["message"]
    assert len(model_state.models) == 1
    assert model_state.models[0].file_name == "model.gguf"