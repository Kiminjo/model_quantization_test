import os
from typing import List, Tuple, Callable, Any

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from config import dir_name_to_class

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class CustomImageFolder(ImageFolder):
    def __init__(self, root: str, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, loader: Callable[[str], Any] = default_loader, is_valid_file: Callable[[str], bool] | None = None):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        
        # 클래스 이름 재매핑
        original_classes = self.classes
        self.classes = [dir_name_to_class[c] for c in original_classes]
        self.class_to_idx = {dir_name_to_class[k]: v for k, v in self.class_to_idx.items()}

    def find_classes(self, directory: str) -> Tuple[List[str], dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    @staticmethod
    def is_valid_image_file(filename: str) -> bool:
        return filename.lower().endswith(IMG_EXTENSIONS)
