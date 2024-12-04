from typing import Dict
from starlette.websockets import WebSocket, WebSocketDisconnect
import asyncio
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_states: Dict[str, bool] = {}  # True if open, False if closed
        self.lock = asyncio.Lock()

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections[client_id] = websocket
            self.connection_states[client_id] = True
        logger.info(f"Client {client_id} connected.")

    async def disconnect(self, client_id: str):
        async with self.lock:
            websocket = self.active_connections.pop(client_id, None)
            self.connection_states.pop(client_id, None)
        if websocket:
            await websocket.close()
            logger.info(f"Client {client_id} disconnected.")

    async def send_message(self, client_id: str, message: str):
        async with self.lock:
            websocket = self.active_connections.get(client_id)
            is_open = self.connection_states.get(client_id, False)
        if websocket and is_open:
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                logger.warning(f"WebSocketDisconnect: Client {client_id} disconnected unexpectedly.")
                await self.disconnect(client_id)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                await self.disconnect(client_id)