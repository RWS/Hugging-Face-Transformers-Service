import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import gc
import asyncio
import os
import shutil

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
from api import model_state  # to access mounted models

client = TestClient(app)

# -------------------------------
# Unit Tests for /v1/model/delete
# -------------------------------
@patch("api.shutil.rmtree")
@patch("api.os.path.exists")
def test_delete_model_success(mock_exists, mock_rmtree):
    """
    Test successful deletion of a mounted model.

    - Prepares a fake model in model_state.
    - Mocks os.path.exists to return True.
    - Mocks shutil.rmtree to simulate filesystem deletion.
    - Asserts HTTP 200 response, removal from state, and rmtree called.
    """
    model_state.models.clear()
    fake_model = MagicMock()
    fake_model.model_name = "test-model"
    fake_model.model = MagicMock()
    fake_model.model.device.type = "cpu"
    fake_model.tokenizer = MagicMock()
    fake_model.pipeline = MagicMock()
    model_state.models.append(fake_model)

    mock_exists.return_value = True
    mock_rmtree.return_value = None

    response = client.delete("/v1/model/delete?model_name=test-model")
    assert response.status_code == 200
    data = response.json()
    assert "has been deleted successfully" in data["message"]
    assert fake_model not in model_state.models
    mock_rmtree.assert_called_once()


@patch("api.os.path.exists")
def test_delete_model_not_found(mock_exists):
    """
    Test deletion when model directory does not exist.

    - Mocks os.path.exists to return False.
    - Expects HTTP 404 response with appropriate error message.
    """
    model_state.models.clear()
    mock_exists.return_value = False

    response = client.delete("/v1/model/delete?model_name=nonexistent-model")
    assert response.status_code == 404
    data = response.json()
    assert "does not exist" in data["detail"]


@patch("api.os.path.exists")
@patch("api.shutil.rmtree", side_effect=Exception("rmtree failed"))
def test_delete_model_rmtree_error(mock_rmtree, mock_exists):
    """
    Test deletion failure due to filesystem error.

    - Mocks shutil.rmtree to raise an exception.
    - Expects HTTP 500 response with appropriate error message.
    """
    model_state.models.clear()
    mock_exists.return_value = True

    response = client.delete("/v1/model/delete?model_name=test-model")
    assert response.status_code == 500
    data = response.json()
    assert "An error occurred while deleting the model" in data["detail"]