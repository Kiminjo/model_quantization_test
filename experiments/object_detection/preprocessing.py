"""
Object Detection용 데이터를 Ultralytics YOLO 포맷에 맞게 변환하는 스크립트

- TIF 파일을 패치 단위로 슬라이딩하며 분할
- JSON 어노테이션을 YOLO TXT 형식으로 변환

Author: Injo Kim
Date: 2025-08-21
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import shutil
from typing import List, Dict

from PIL import Image, ImageDraw
import multiprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 프로젝트 루트를 sys.path에 추가 (상대 경로 import를 위함)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from object_detection.config import (
    CLASSES,
    CLASS_TO_IDX,
    PATCH_SIZES,
    OVERLAP_RATIO
)

def setup_logger() -> logging.Logger:
    """로거를 설정합니다."""
    log_dir = project_root.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_filename = f"obj_preprocessing_{datetime.now().strftime('%Y%m%d')}.log"
    log_filepath = log_dir / log_filename
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

logger = setup_logger()

def split_units(
    raw_data_dir: Path, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
) -> Dict[str, List[Dict[str, Path]]]:
    """
    TIF/JSON 유닛을 train, val, test 세트로 분할합니다.
    TIF 파일 기준으로 분할하며, 매칭되는 JSON 파일이 없는 경우 경고를 출력합니다.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if not (train_ratio + val_ratio + test_ratio) == 1.0:
        logger.error("Ratios must sum to 1.0")
        raise ValueError("Ratios must sum to 1.0")

    # rglob으로 하위 디렉토리까지 모두 검색
    tif_files = sorted(list(raw_data_dir.rglob('*.tif')))
    
    units = []
    for tif_path in tif_files:
        json_path = tif_path.with_suffix('.json')
        if json_path.exists():
            units.append({'image': tif_path, 'label': json_path})
        else:
            logger.warning(f"Label file not found for {tif_path}. Skipping this unit.")
    
    if not units:
        logger.error("No valid TIF/JSON units found in the raw data directory.")
        raise FileNotFoundError("No valid TIF/JSON units found.")

    # 1차 분할: train vs (val + test)
    train_units, remaining_units = train_test_split(
        units, train_size=train_ratio, random_state=42, shuffle=True
    )

    # 2차 분할: val vs test
    if remaining_units:
        val_size = val_ratio / (val_ratio + test_ratio)
        val_units, test_units = train_test_split(
            remaining_units, train_size=val_size, random_state=42, shuffle=True
        )
    else:
        val_units, test_units = [], []
    
    logger.info(f"Data split: {len(train_units)} train, {len(val_units)} val, {len(test_units)} test units.")
    
    return {'train': train_units, 'val': val_units, 'test': test_units}

def convert_bbox_to_yolo(
    bbox: List[int], patch_w: int, patch_h: int, class_idx: int
) -> str:
    """바운딩 박스를 YOLO 형식으로 변환합니다."""
    x_min, y_min, x_max, y_max = bbox
    
    # 바운딩 박스의 중심점, 너비, 높이 계산
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # 패치 크기로 정규화
    x_center /= patch_w
    y_center /= patch_h
    width /= patch_w
    height /= patch_h
    
    # 값들이 0과 1 사이에 있도록 보정
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def draw_and_save_visualization(
    image_path: Path,
    label_path: Path,
    viz_save_path: Path
):
    """저장된 패치와 라벨을 이용해 시각화 결과를 생성합니다."""
    try:
        image = Image.open(image_path).convert("RGB")
        patch_w, patch_h = image.size
        draw = ImageDraw.Draw(image)

        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            x_center, y_center, width, height = map(float, parts[1:])
            
            abs_x_center = x_center * patch_w
            abs_y_center = y_center * patch_h
            abs_width = width * patch_w
            abs_height = height * patch_h
            
            x_min = abs_x_center - (abs_width / 2)
            y_min = abs_y_center - (abs_height / 2)
            x_max = abs_x_center + (abs_width / 2)
            y_max = abs_y_center + (abs_height / 2)
            
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
            
        viz_save_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(viz_save_path)

    except Exception as e:
        print(f"Could not visualize {image_path.name}: {e}")

def process_unit(
    unit: Dict[str, Path],
    patch_size: int,
    overlap_ratio: float,
    save_dir: Path,
    visualize: bool,
    viz_save_dir: Path
):
    """단일 유닛(이미지/라벨)을 처리하여 패치를 생성하고 저장합니다."""
    try:
        image_path = unit['image']
        label_path = unit['label']

        image = Image.open(image_path)
        img_w, img_h = image.size

        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        annotations = label_data.get('annotations', [])
        if not annotations:
            return

        images_save_dir = save_dir / 'images'
        labels_save_dir = save_dir / 'labels'
        images_save_dir.mkdir(parents=True, exist_ok=True)
        labels_save_dir.mkdir(parents=True, exist_ok=True)

        stride = int(patch_size * (1 - overlap_ratio))
        
        patch_count = 0
        for y_start in range(0, img_h, stride):
            for x_start in range(0, img_w, stride):
                y_end = min(y_start + patch_size, img_h)
                x_end = min(x_start + patch_size, img_w)
                
                # 패치 크기가 설정값보다 작으면 건너뜀 (가장자리 처리)
                if y_end - y_start != patch_size or x_end - x_start != patch_size:
                    continue

                yolo_labels = []
                
                for ann in annotations:
                    label = ann.get('label')
                    if label not in CLASS_TO_IDX:
                        continue
                        
                    class_idx = CLASS_TO_IDX[label]
                    bbox = ann.get('bbox') # [x_min, y_min, x_max, y_max]
                    if not bbox or len(bbox) != 4:
                        continue
                    
                    obj_x_min, obj_y_min, obj_x_max, obj_y_max = bbox

                    # 패치와 객체 바운딩 박스의 교차 영역 계산
                    inter_x_min = max(x_start, obj_x_min)
                    inter_y_min = max(y_start, obj_y_min)
                    inter_x_max = min(x_end, obj_x_max)
                    inter_y_max = min(y_end, obj_y_max)

                    # 교차 영역이 유효한지(넓이가 있는지) 확인
                    if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                        # 패치 내 로컬 좌표로 변환
                        local_x_min = inter_x_min - x_start
                        local_y_min = inter_y_min - y_start
                        local_x_max = inter_x_max - x_start
                        local_y_max = inter_y_max - y_start

                        # 변환된 바운딩 박스가 유의미한 크기인지 확인 (1x1 초과)
                        if (local_x_max - local_x_min) > 1 and (local_y_max - local_y_min) > 1:
                            yolo_str = convert_bbox_to_yolo(
                                [local_x_min, local_y_min, local_x_max, local_y_max],
                                patch_size, patch_size, class_idx
                            )
                            yolo_labels.append(yolo_str)

                if yolo_labels:
                    patch_count += 1
                    patch_image = image.crop((x_start, y_start, x_end, y_end))
                    
                    base_filename = f"{image_path.stem}_p{patch_size}_{patch_count}"
                    img_save_path = images_save_dir / f"{base_filename}.png"
                    lbl_save_path = labels_save_dir / f"{base_filename}.txt"

                    # Pillow 이미지를 저장
                    patch_image.save(img_save_path)

                    with open(lbl_save_path, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(yolo_labels))

                    if visualize:
                        viz_img_save_path = viz_save_dir / f"{base_filename}.png"
                        draw_and_save_visualization(img_save_path, lbl_save_path, viz_img_save_path)

    except Exception as e:
        logger.error(f"Failed to process unit {unit.get('image', 'N/A').name}: {e}", exc_info=True)

def create_yolo_yaml(processed_data_dir: Path):
    """YOLOv8 데이터셋 YAML 파일을 생성합니다."""
    class_names_str = "\n".join([f"  {i}: {name}" for i, name in enumerate(CLASSES)])

    yaml_content = f"""path: {processed_data_dir.resolve()}
train: train/images
val: val/images
test: test/images

nc: {len(CLASSES)}
names:
{class_names_str}
"""
    yaml_path = processed_data_dir / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    logger.info(f"YOLOv8 YAML file created at: {yaml_path}")

def main(visualize: bool = True):
    """메인 실행 함수"""
    raw_data_dir = project_root.parent / 'data' / 'raw' / 'object_detection'
    processed_data_dir = project_root.parent / 'data' / 'processed' / 'object_detection'
    
    # 기존 디렉토리 삭제 후 재생성
    if processed_data_dir.exists():
        logger.info(f"Removing existing directory: {processed_data_dir}")
        shutil.rmtree(processed_data_dir)
    logger.info(f"Creating new directory: {processed_data_dir}")
    processed_data_dir.mkdir(parents=True)

    if visualize:
        viz_dir = processed_data_dir / 'viz'
        if viz_dir.exists():
            shutil.rmtree(viz_dir)
        logger.info(f"Creating new directory for visualizations: {viz_dir}")
        viz_dir.mkdir(parents=True)

    logger.info("Preprocessing started.")
    
    # 1. 데이터셋 분할
    unit_splits = split_units(raw_data_dir)

    # 2. 멀티프로세싱을 위한 작업 목록 생성
    tasks = []
    for split_name, units in unit_splits.items():
        if not units:
            logger.info(f"No units to process for '{split_name}' split. Skipping.")
            continue

        split_save_dir = processed_data_dir / split_name
        viz_save_dir = processed_data_dir / 'viz' / split_name
        
        for unit in units:
            for patch_size in PATCH_SIZES:
                tasks.append((unit, patch_size, OVERLAP_RATIO, split_save_dir, visualize, viz_save_dir))
    
    if not tasks:
        logger.warning("No tasks to process. Exiting.")
        return

    # 3. 멀티프로세싱 풀을 사용하여 병렬 처리
    num_processes = multiprocessing.cpu_count()
    logger.info(f"Using {num_processes} processes for preprocessing...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.starmap(process_unit, tasks), total=len(tasks), desc="Processing units"))
    
    # 4. YOLO YAML 파일 생성
    logger.info("Creating YOLOv8 dataset YAML file...")
    create_yolo_yaml(processed_data_dir)
    
    logger.info("Preprocessing finished.")

if __name__ == '__main__':
    main()
