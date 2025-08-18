"""
Ultralytics YOLOv8 Classification 모델을 학습하는 스크립트

Author: Injo Kim
Date: 2025-08-18
"""

import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).resolve().parent.parent
    
    # 전처리된 데이터 경로 설정
    processed_data_path = project_root / 'data' / 'processed' / 'classification'

    # 데이터셋이 준비되었는지 확인
    if not processed_data_path.exists() or not (processed_data_path / 'train').exists():
        print(f"Error: Processed data not found at '{processed_data_path}'")
        print("Please run 'classification/preprocessing.py' first.")
        sys.exit(1)

    # 사전 학습된 YOLOv11n-cls 모델 로드
    model = YOLO('yolo11n-cls.pt')

    # 모델 학습
    results = model.train(
        data=str(processed_data_path),
        epochs=100,
        imgsz=400,
        device=3,
        project='runs',
        name='classification_experiment'
    )
    
    print("Training complete.")
    print(f"Results saved to {results.save_dir}")

if __name__ == '__main__':
    main()