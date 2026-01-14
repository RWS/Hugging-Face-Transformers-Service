import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

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
# Unit Tests for v1/model/unmount
# -------------------------------

@patch("api.torch.cuda.empty_cache")
def test_unmount_model_gpu(mock_empty_cache):
    """Unmount a model on CUDA device and ensure GPU cache is cleared."""
    model_state.device = "cuda"

    fake_model = MagicMock()
    fake_model.model_name = "cuda-model"

    from types import SimpleNamespace
    fake_model.model = MagicMock()
    fake_model.model.device = SimpleNamespace(type="cuda")
    fake_model.model.cpu = MagicMock()
    fake_model.tokenizer = MagicMock()
    fake_model.pipeline = MagicMock()

    model_state.models.append(fake_model)

    response = client.post("/v1/model/unmount", json={"model_name": "cuda-model"})

    assert response.status_code == 200
    mock_empty_cache.assert_called_once()
    assert fake_model not in model_state.models

@patch("gc.collect")
def test_unmount_model_gc_called(mock_gc):
    """Ensure garbage collector is invoked during unmount."""
    fake_model = MagicMock()
    fake_model.model_name = "gc-model"
    fake_model.model = MagicMock()
    fake_model.tokenizer = MagicMock()
    fake_model.pipeline = MagicMock()
    model_state.models.append(fake_model)

    response = client.post("/v1/model/unmount", json={"model_name": "gc-model"})

    assert response.status_code == 200
    mock_gc.assert_called_once()
    assert fake_model not in model_state.models