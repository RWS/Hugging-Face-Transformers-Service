import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import pytest

# -------------------------------
# Make src discoverable
# Add src folder to path so that imports from src work in tests
# -------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from app import app
from api import manager, model_states, fine_tune_model

client = TestClient(app)

# -------------------------------
# Fixtures
# -------------------------------

@pytest.fixture(autouse=True)
def setup_manager():
    """Clear active connections and model_states before each test."""
    manager.active_connections.clear()
    model_states.clear()

# -------------------------------
# Helper functions
# -------------------------------
def get_valid_payload(client_id="client1"):
    return {
        "client_id": client_id,
        "model_path": "C:/HuggingFace/Models/test-model",
        "output_dir": "C:/HuggingFace/Models/test-model-finetuned",
        "data_file": "C:/HuggingFace/Data/data.csv",
        "validation_file": "C:/HuggingFace/Data/validation.csv",
        "source_lang": "en_XX",
        "target_lang": "it_IT",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "max_length": 512,
        "save_strategy": "steps",
        "save_steps": 10,
        "save_total_limit": 2
    }

# -------------------------------
# Unit Tests for /v1/model/fine-tune
# -------------------------------

def test_fine_tune_client_not_connected():
    """
    Test /v1/model/fine-tune returns 400 if client_id is not connected.

    The endpoint requires an active WebSocket connection for the client.
    """
    payload = get_valid_payload(client_id="nonexistent_client")
    response = client.post("/v1/model/fine-tune", json=payload)
    assert response.status_code == 400
    assert "not connected" in response.json()["detail"]


@patch("api.BackgroundTasks.add_task")
@patch("api.fine_tune_model")
def test_fine_tune_starts_background_task(mock_fine_tune_model, mock_add_task):
    """
    Test that a fine-tune request starts a background task for a connected client.

    - Adds the client to active_connections.
    - Sends a POST request to /v1/model/fine-tune.
    - Verifies the background task is added with the correct function.
    """
    from api import manager

    client_id = "client1"
    manager.active_connections[client_id] = MagicMock()

    payload = get_valid_payload(client_id=client_id)
    response = client.post("/v1/model/fine-tune", json=payload)

    assert response.status_code == 200
    mock_add_task.assert_called_once_with(mock_fine_tune_model, ANY)


def test_fine_tune_cancel_requested_initial_state():
    """
    Test that cancel_requested flag is False initially for a new fine-tune request.

    - Ensures the model state for the client starts with cancel_requested = False.
    - Verifies HTTP 200 response.
    """
    manager.active_connections["client2"] = MagicMock()
    payload = get_valid_payload(client_id="client2")

    with patch("fastapi.BackgroundTasks.add_task"):
        response = client.post("/v1/model/fine-tune", json=payload)

    assert response.status_code == 200
    assert model_states["client2"]["cancel_requested"] is False