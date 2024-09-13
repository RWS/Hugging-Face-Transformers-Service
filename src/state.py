from typing import List
import torch
from models import LocalModel 

class DownloadState:
    def __init__(self):
        self.is_downloading = False
        self.download_progress = []
        self.models: List[LocalModel] = []  # List to hold the LocalModel instances
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create a singleton instance
download_state = DownloadState()