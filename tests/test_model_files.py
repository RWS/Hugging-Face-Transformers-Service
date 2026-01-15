from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

# -------------------------------
# Unit Tests for /v1/model/files
# -------------------------------
def test_model_files_empty_model_name():
    """
        Test that an empty model_name query parameter returns HTTP 400.

        - Ensures user provides a model_name.
    """
    response = client.get("/v1/model/files?model_name=")
    assert response.status_code == 400
    assert "`model_name` must be provided" in response.json()["detail"]

@patch("api.fetch_model_info")
def test_model_files_no_siblings(mock_fetch_info):
    """
        Test behavior when fetch_model_info returns an object without 'siblings'.

        - Should return HTTP 400 with an appropriate error message.
    """
    mock_obj = MagicMock()
    del mock_obj.siblings  # remove siblings attribute
    mock_fetch_info.return_value = mock_obj

    response = client.get("/v1/model/files?model_name=test-model")
    assert response.status_code == 400
    assert "Error accessing model files" in response.json()["detail"]

@patch("api.fetch_model_info")
@patch("api.hf_hub_url")
@patch("api.get_file_size_via_head", new_callable=AsyncMock)
@patch("api.get_file_size_via_get", new_callable=AsyncMock)
def test_model_files_root_and_non_root(mock_get_get, mock_get_head, mock_hub_url, mock_fetch_info):
    """
        Test filtering of root vs non-root files.

        - Only files at the root of the repo should appear in the response.
    """
    file_root = MagicMock()
    file_root.rfilename = "config.json"
    file_root.size = 1024

    file_non_root = MagicMock()
    file_non_root.rfilename = "subdir/model.bin"
    file_non_root.size = 2048

    info_mock = MagicMock()
    info_mock.siblings = [file_root, file_non_root]
    mock_fetch_info.return_value = info_mock

    mock_hub_url.side_effect = lambda repo_id, filename: f"https://fake/{filename}"
    mock_get_head.return_value = None
    mock_get_get.return_value = None

    response = client.get("/v1/model/files?model_name=test-model")
    assert response.status_code == 200
    data = response.json()
    # Only root file should appear
    assert len(data["files"]) == 1
    assert data["files"][0]["file_name"] == "config.json"

@patch("api.asyncio.to_thread", new_callable=AsyncMock)
@patch("api.hf_hub_url")
@patch("api.get_file_size_via_head", new_callable=AsyncMock)
@patch("api.get_file_size_via_get", new_callable=AsyncMock)
def test_model_files_size_unknown(mock_get_get, mock_get_head, mock_hub_url, mock_fetch_info):
    """
        Test handling of files with unknown size.

        - If size is None, the API should report "Unknown" as file_size.
    """
    file_no_size = MagicMock()
    file_no_size.rfilename = "unknown.bin"
    file_no_size.size = None

    info_mock = MagicMock()
    info_mock.siblings = [file_no_size]
    mock_fetch_info.return_value = info_mock

    mock_hub_url.side_effect = lambda repo_id, filename: f"https://fake/{filename}"
    mock_get_head.return_value = None
    mock_get_get.return_value = None

    response = client.get("/v1/model/files?model_name=test-model")
    assert response.status_code == 200
    data = response.json()
    assert data["files"][0]["file_size"] == "Unknown"

@patch("api.asyncio.to_thread", new_callable=AsyncMock)
def test_model_files_api_key_used(mock_to_thread):
    """
        Test that an API key provided in the request header is passed to fetch_model_info.

        - Ensures correct integration with asyncio.to_thread for async execution.
    """
    file_root = MagicMock()
    file_root.rfilename = "file.txt"
    file_root.size = 100

    info_mock = MagicMock()
    info_mock.siblings = [file_root]
    mock_to_thread.return_value = info_mock

    response = client.get(
        "/v1/model/files?model_name=test-model",
        headers={"api-key": "dummy-key"}  # ✅ correct header
    )

    assert response.status_code == 200
    data = response.json()
    assert data["files"][0]["file_name"] == "file.txt"

    called_args, _ = mock_to_thread.call_args
    func = called_args[0]
    args = called_args[1:]
    assert func.__name__ == "fetch_model_info"
    assert args[1] == "dummy-key"

@patch("api.asyncio.to_thread", new_callable=AsyncMock)
def test_model_files_skip_no_rfilename(mock_to_thread):
    """
        Test that files with no rfilename are skipped.

        - Ensures response only includes files with valid rfilename.
    """
    file_no_name = MagicMock()
    file_no_name.rfilename = None
    file_no_name.size = 100

    info_mock = MagicMock()
    info_mock.siblings = [file_no_name]

    mock_to_thread.return_value = info_mock

    response = client.get("/v1/model/files?model_name=test-model")
    assert response.status_code == 200
    data = response.json()

    # The file should be skipped because rfilename is None
    assert len(data["files"]) == 0

@patch("api.fetch_model_info")
@patch("api.hf_hub_url")
@patch("api.get_file_size_via_head", new_callable=AsyncMock)
@patch("api.get_file_size_via_get", new_callable=AsyncMock)
def test_model_files_root_files(mock_get_get, mock_get_head, mock_hub_url, mock_fetch_info):
    """
        Test multiple root files and fallback to HEAD request for unknown sizes.

        - Both files should appear in the response.
        - Fallback logic is triggered for files with unknown size.
    """
    file1 = MagicMock()
    file1.rfilename = "config.json"
    file1.size = 1024
    file2 = MagicMock()
    file2.rfilename = "pytorch_model.bin"
    file2.size = None

    info_mock = MagicMock()
    info_mock.siblings = [file1, file2]
    mock_fetch_info.return_value = info_mock

    mock_hub_url.side_effect = lambda repo_id, filename: f"https://fake/{filename}"
    mock_get_head.return_value = 2048
    mock_get_get.return_value = None

    response = client.get("/v1/model/files?model_name=test-model")
    assert response.status_code == 200
    data = response.json()
    assert len(data["files"]) == 2
    assert data["files"][0]["file_name"] == "config.json"
    assert data["files"][1]["file_size"].startswith("2")  # head fallback

@patch("api.fetch_model_info", side_effect=Exception("Network error"))
def test_model_files_fetch_info_exception(mock_fetch_info):
    """
        Test exception handling in fetch_model_info.

        - If fetch_model_info raises an exception, API should return HTTP 400 with the error message.
    """
    response = client.get("/v1/model/files?model_name=test-model")
    assert response.status_code == 400
    assert "Network error" in response.json()["detail"]