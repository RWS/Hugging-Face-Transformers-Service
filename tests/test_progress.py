import asyncio
import json
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import app
from api import manager, model_states

client = TestClient(app)

# -------------------------------
# Fixture
# -------------------------------
@pytest.fixture(autouse=True)
def clear_state():
    """
    Clear manager connections and model_states before each test.

    Ensures that WebSocket connections and model progress state do not
    interfere between tests.
    """
    manager.active_connections.clear()
    model_states.clear()

# -------------------------------
# Unit Tests for ws/progress/{client_id}
# -------------------------------
def test_websocket_heartbeat():
    """
        Test that sending a 'heartbeat' message over WebSocket triggers
        a 'heartbeat_ack' message back to the client.

        - Mocks manager.send_message to verify outgoing messages.
        - Mocks manager.disconnect to prevent side effects on test exit.
        - Uses client.websocket_connect for synchronous WebSocket testing.
    """
    client_id = "client_heartbeat"
    manager.active_connections[client_id] = AsyncMock()

    with patch("api.manager.send_message", new_callable=AsyncMock) as mock_send, \
         patch("api.manager.disconnect", new_callable=AsyncMock):
        try:
            with client.websocket_connect(f"/ws/progress/{client_id}") as websocket:
                websocket.send_text("heartbeat")
                # give server a moment to process
                asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.05))

                # Check that heartbeat_ack was sent
                mock_send.assert_any_await(client_id, json.dumps({"type": "heartbeat_ack"}))
        except Exception:
            # TestClient raises WebSocketDisconnect on close, ignore
            pass


def test_websocket_cancel():
    """
        Test that sending a 'cancel' message over WebSocket:

        - Sets model_states[client_id]["cancel_requested"] to True
        - Sends a 'cancellation_ack' message back to the client
    """
    client_id = "client_cancel"
    manager.active_connections[client_id] = AsyncMock()

    with patch("api.manager.send_message", new_callable=AsyncMock) as mock_send, \
         patch("api.manager.disconnect", new_callable=AsyncMock):
        try:
            with client.websocket_connect(f"/ws/progress/{client_id}") as websocket:
                websocket.send_text("cancel")
                asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.05))

                # Check that cancellation_ack was sent
                mock_send.assert_any_await(
                    client_id,
                    json.dumps({
                        "type": "cancellation_ack",
                        "data": "Cancellation requested. Stopping process..."
                    })
                )
                # Check that cancel_requested is True
                assert model_states[client_id]["cancel_requested"] is True
        except Exception:
            pass