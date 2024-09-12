from typing import List
from models import LocalModel 

class DownloadState:
    def __init__(self):
        self.is_downloading = False
        self.download_progress = []
        self.models: List[LocalModel] = []  # List to hold the LocalModel instances

# Create a singleton instance
download_state = DownloadState()