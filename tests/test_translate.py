import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import uuid

# -------------------------------
# Make src discoverable
# Add src folder to path so that imports from src work in tests
# -------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi.testclient import TestClient
from app import app
from api import model_state
from models import TranslationRequest, TranslationResponse  # Assuming your Pydantic models are here

client = TestClient(app)

# -------------------------------
# Unit Tests for v1/translation
# -------------------------------

def test_translation_model_not_mounted():
    """Return 400 if model is not mounted."""
    model_state.models.clear()
    response = client.post("/v1/translation", json={"model_name": "nonexistent", "text": "Hello"})
    assert response.status_code == 400
    assert "not currently mounted" in response.json()["detail"]


def test_translation_pipeline_none():
    """Return 400 if model pipeline is None."""
    model_state.models.clear()
    fake_model = MagicMock()
    fake_model.model_name = "test-model"
    fake_model.pipeline = None
    model_state.models.append(fake_model)

    response = client.post("/v1/translation", json={"model_name": "test-model", "text": "Hello"})
    assert response.status_code == 400
    assert "not currently mounted" in response.json()["detail"]


def test_translation_success():
    """Return successful translation when pipeline works normally."""
    model_state.models.clear()
    fake_pipeline = MagicMock(return_value=[{"translation_text": "Bonjour"}])
    fake_model = MagicMock()
    fake_model.model_name = "test-model"
    fake_model.pipeline = fake_pipeline
    model_state.models.append(fake_model)

    response = client.post("/v1/translation", json={"model_name": "test-model", "text": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["generated_text"] == "Bonjour"
    assert data["model"] == "test-model"
    assert "id" in data


def test_translation_unexpected_pipeline_output():
    """Pipeline returns unexpected format -> generated_text should be empty string."""
    model_state.models.clear()
    fake_pipeline = MagicMock(return_value=[{"wrong_key": "value"}])
    fake_model = MagicMock()
    fake_model.model_name = "test-model"
    fake_model.pipeline = fake_pipeline
    model_state.models.append(fake_model)

    response = client.post("/v1/translation", json={"model_name": "test-model", "text": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["generated_text"] == ""


def test_translation_pipeline_raises_exception():
    """If the pipeline raises an exception, return HTTP 500."""
    model_state.models.clear()
    fake_pipeline = MagicMock(side_effect=Exception("Pipeline error"))
    fake_model = MagicMock()
    fake_model.model_name = "test-model"
    fake_model.pipeline = fake_pipeline
    model_state.models.append(fake_model)

    response = client.post("/v1/translation", json={"model_name": "test-model", "text": "Hello"})
    assert response.status_code == 500
    assert "Error during translation" in response.json()["detail"]


def test_translation_empty_text():
    """Empty input text should still return 200 with empty generated_text."""
    model_state.models.clear()
    fake_pipeline = MagicMock(return_value=[{"translation_text": ""}])
    fake_model = MagicMock()
    fake_model.model_name = "test-model"
    fake_model.pipeline = fake_pipeline
    model_state.models.append(fake_model)

    response = client.post("/v1/translation", json={"model_name": "test-model", "text": ""})
    assert response.status_code == 200
    data = response.json()
    assert data["generated_text"] == ""