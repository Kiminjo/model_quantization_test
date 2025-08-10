import os
from typing import List, Tuple
from torchvision.datasets import ImageFolder

from config import dir_name_to_class


class CustomImageFolder(ImageFolder):
    def find_classes(self, directory: str) -> Tuple[List[str], dict[str, int]]:
        class_to_idx = {dir_name_to_class[c]: i for i, c in enumerate(os.listdir(directory)) if os.path.isdir(os.path.join(directory, c))}
        
        sorted_class_to_idx = {k: v for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])}
        
        if not sorted_class_to_idx:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        return list(sorted_class_to_idx.keys()), sorted_class_to_idx
