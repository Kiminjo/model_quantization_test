"""
Ultralytics YOLOv8 Classification 모델을 학습하는 스크립트

Author: Injo Kim
Date: 2025-08-18
"""

import sys
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

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

    # --- 학습 파라미터 및 동적 실험 이름 설정 ---
    model_path = 'yolo11n-cls.pt'
    img_size = 400
    epochs = 100
    device = 3
    
    model_name = Path(model_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"baseline_{model_name}_imgsz{img_size}_{timestamp}"
    
    # 사전 학습된 YOLOv11n-cls 모델 로드
    model = YOLO(model_path)

    # 모델 학습
    results = model.train(
        data=str(processed_data_path),
        epochs=epochs,
        imgsz=img_size,
        device=device,
        project='runs',
        name=experiment_name
    )
    
    print("Training complete.")
    print(f"Results saved to {results.save_dir}")

    # --- TensorRT로 모델 변환 (INT8 Quantization) ---
    print("\nExporting the best model to TensorRT with INT8 quantization...")
    
    # 가장 성능이 좋은 모델('best.pt') 로드
    best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
    if best_model_path.exists():
        best_model = YOLO(best_model_path)
        
        # TensorRT로 변환
        # int8=True 옵션은 Post-Training Quantization(PTQ)을 적용합니다.
        exported_model_path = best_model.export(format='tensorrt', int8=True)
        
        print(f"Successfully exported model to: {exported_model_path}")
    else:
        print(f"Error: Could not find the best model at {best_model_path}")


if __name__ == '__main__':
    main()