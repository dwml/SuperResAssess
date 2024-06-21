from pathlib import Path
from typing import Optional

class HCP:
    def __init__(self, data_dir: Path, download: Optional[bool] = False):
        self.data_dir = data_dir
        self.download = download
