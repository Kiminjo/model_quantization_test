"""
Hubble Format의 Classifcation용 데이터를 Ultralytics Classification 포맷에 맞게 변환하는 스크립트 

Author: Injo Kim 
Date: 2025-08-18
"""

import sys
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가하여 config.py를 임포트합니다.
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    from config import dir_name_to_class
except ImportError:
    print("Error: config.py not found. Make sure it is in the project root directory.")
    sys.exit(1)

# 로거 설정
def setup_logger():
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_filename = f"cls_preprocessing_{datetime.now().strftime('%Y%m%d')}.log"
    log_filepath = log_dir / log_filename
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 핸들러 중복 추가 방지
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # 파일 핸들러
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # 콘솔 핸들러
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logger()

def preprocess_and_split_data(
    raw_data_dir: Path, 
    processed_data_dir: Path, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1
):
    """
    원본 데이터를 train, val, test 세트로 분할하고 지정된 경로에 복사합니다.
    """
    # 비율 합계 검증
    if not (train_ratio + val_ratio + test_ratio) == 1.0:
        logger.error("Ratios must sum to 1.0")
        raise ValueError("Ratios must sum to 1.0")

    # 기존 디렉토리 삭제 후 새로 생성
    if processed_data_dir.exists():
        logger.info(f"Removing existing directory: {processed_data_dir}")
        shutil.rmtree(processed_data_dir)
    
    logger.info(f"Creating new directory structure at: {processed_data_dir}")
    
    # train, val, test 폴더 생성
    train_dir = processed_data_dir / 'train'
    val_dir = processed_data_dir / 'val'
    test_dir = processed_data_dir / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp')

    logger.info("Starting data splitting and copying process...")
    
    # 각 클래스별로 파일 분할 및 복사
    for dir_name, class_name in dir_name_to_class.items():
        class_dir = raw_data_dir / dir_name
        if not class_dir.is_dir():
            logger.warning(f"Directory '{class_dir}' not found. Skipping.")
            continue

        # rglob을 사용하여 하위 디렉토리까지 재귀적으로 이미지 파일 검색
        images = [f for ext in IMG_EXTENSIONS for f in class_dir.glob(f'**/*.{ext}')]
        
        if not images:
            logger.warning(f"No images found in '{class_dir}' and its subdirectories. Skipping.")
            continue

        # 1차 분할: train vs (val + test)
        train_images, remaining_images = train_test_split(
            images, train_size=train_ratio, random_state=42, shuffle=True
        )

        # 2차 분할: val vs test
        # 남은 데이터 내에서의 비율 계산
        val_size = val_ratio / (val_ratio + test_ratio)
        val_images, test_images = train_test_split(
            remaining_images, train_size=val_size, random_state=42, shuffle=True
        )
        
        # 각 세트의 클래스 폴더 생성
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)
        (test_dir / class_name).mkdir(exist_ok=True)
        
        # 파일 복사
        for img_path in train_images:
            shutil.copy(img_path, train_dir / class_name / img_path.name)
        for img_path in val_images:
            shutil.copy(img_path, val_dir / class_name / img_path.name)
        for img_path in test_images:
            shutil.copy(img_path, test_dir / class_name / img_path.name)
            
        logger.info(f"Processed class '{class_name}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test.")
        
    logger.info("\nData preprocessing complete.")


if __name__ == '__main__':
    # 경로 설정
    raw_data_path = project_root / 'data' / 'raw' / 'classification'
    processed_data_path = project_root / 'data' / 'processed' / 'classification'
    
    preprocess_and_split_data(raw_data_path, processed_data_path)
