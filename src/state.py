from typing import List
import torch
from models import LocalModel 
import os

class ModelState:
    def __init__(self):
        self.is_downloading = False
        self.download_progress = []
        self.models: List[LocalModel] = []  # List to hold the LocalModel instances
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cpu_cores = os.cpu_count() or 1

# singleton instance
model_state = ModelState()