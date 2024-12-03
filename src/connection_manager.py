from typing import Dict
from fastapi import WebSocket
#import asyncio

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    async def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send_message(self, client_id: str, message: str):
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_text(message)
            
# class ConnectionManager:
#     def __init__(self):
#         self.active_connections: Dict[str, WebSocket] = {}
#         self.lock = asyncio.Lock()

#     async def connect(self, client_id: str, websocket: WebSocket):
#         await websocket.accept()
#         async with self.lock:
#             self.active_connections[client_id] = websocket

#     async def disconnect(self, client_id: str):
#         async with self.lock:
#             if client_id in self.active_connections:
#                 del self.active_connections[client_id]

#     async def send_message(self, client_id: str, message: str):
#         async with self.lock:
#             websocket = self.active_connections.get(client_id)
#             if websocket:
#                 try:
#                     await websocket.send_text(message)
#                 except Exception as e:
#                     print(f"Error sending message to {client_id}: {e}")