from typing import List
import torch
from src.models import LocalModel 

class ModelState:
    def __init__(self):
        self.is_downloading = False
        self.download_progress = []
        self.models: List[LocalModel] = []  # List to hold the LocalModel instances
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# singleton instance
model_state = ModelState()