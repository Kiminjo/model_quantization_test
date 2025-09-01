"""
Ultralytics YOLOv8 Object Detection 모델을 학습하는 스크립트

Author: Injo Kim
Date: 2025-08-18
"""

import sys
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import shutil
import logging

def setup_logging(project_root: Path):
    """로깅 설정"""
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = log_dir / f"od_train_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Log file will be saved to {log_filename}")

def prepare_fp32_model(original_weights_dir: Path, fp32_dir: Path):
    """FP32 모델을 준비합니다 (기존 weights 폴더 복사)."""
    logging.info(f"Preparing FP32 weights in '{fp32_dir}'")
    if fp32_dir.exists():
        shutil.rmtree(fp32_dir)
    shutil.copytree(original_weights_dir, fp32_dir)
    logging.info("FP32 model preparation complete.")

def prepare_and_export_quantized_model(
    precision: str,
    source_pt_path: Path,
    target_dir: Path,
    export_params: dict
):
    """양자화된 모델(FP16, INT8)을 준비하고 내보냅니다."""
    logging.info(f"--- Starting {precision} model export ---")
    
    # 1. 대상 디렉토리 준비 및 모델 복사
    target_dir.mkdir(parents=True, exist_ok=True)
    target_pt_path = target_dir / 'best.pt'
    shutil.copy2(source_pt_path, target_pt_path)
    logging.info(f"Prepared {precision} weights in '{target_dir}'")

    # 2. 모델 로드 및 내보내기
    logging.info(f"Exporting {precision} model to TensorRT...")
    model = YOLO(target_pt_path)
    exported_path = model.export(format='tensorrt', **export_params)
    logging.info(f"Successfully exported {precision} model to: {exported_path}")

def export_model_variants(results, data_yaml_path):
    """학습된 모델을 FP32, FP16, INT8 버전으로 내보냅니다."""
    logging.info("Starting model export process for FP32, FP16, and INT8...")
    
    save_dir = Path(results.save_dir)
    original_weights_dir = save_dir / 'weights'
    best_pt_path = original_weights_dir / 'best.pt'

    if not best_pt_path.exists():
        logging.error(f"Could not find the best model at {best_pt_path}")
        sys.exit(1)

    # FP32 모델 처리
    fp32_dir = save_dir / 'weights_fp32'
    prepare_fp32_model(original_weights_dir, fp32_dir)
    logging.info("--- Starting FP32 model export ---")
    model_fp32 = YOLO(fp32_dir / 'best.pt')
    fp32_exported_path = model_fp32.export(format='tensorrt')
    logging.info(f"Successfully exported FP32 model to: {fp32_exported_path}")

    # FP16 모델 처리
    fp16_dir = save_dir / 'weights_fp16'
    prepare_and_export_quantized_model(
        precision='FP16',
        source_pt_path=best_pt_path,
        target_dir=fp16_dir,
        export_params={'half': True}
    )

    # INT8 모델 처리
    int8_dir = save_dir / 'weights_int8_ptq'
    prepare_and_export_quantized_model(
        precision='INT8',
        source_pt_path=best_pt_path,
        target_dir=int8_dir,
        export_params={'int8': True, 'data': str(data_yaml_path)}
    )

    logging.info("All models have been exported successfully.")


def main():
    # 프로젝트 루트 경로 설정
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # 로깅 설정
    setup_logging(project_root)
    
    # 전처리된 데이터 경로 설정
    processed_data_path = project_root / 'data' / 'processed' / 'object_detection'
    data_yaml_path = processed_data_path / 'data.yaml'

    # 데이터셋이 준비되었는지 확인
    if not data_yaml_path.exists():
        logging.error(f"data.yaml not found at '{data_yaml_path}'")
        logging.error("Please run 'object_detection/preprocessing.py' first.")
        sys.exit(1)

    # --- 학습 파라미터 및 동적 실험 이름 설정 ---
    model_path = 'yolo11n.pt'  # 객체 탐지 모델로 변경
    img_size = 400
    epochs = 100
    device = 3
    
    model_name = Path(model_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"baseline_{model_name}_imgsz{img_size}_{timestamp}"
    
    logging.info("--- Starting Model Training ---")
    logging.info(f"Model: {model_path}, Image Size: {img_size}, Epochs: {epochs}, Device: {device}")
    logging.info(f"Experiment name: {experiment_name}")

    # 사전 학습된 YOLOv8n 모델 로드
    model = YOLO(model_path)

   # 모델 학습
    results = model.train(
        data=str(data_yaml_path),  # data.yaml 파일 경로 지정
        epochs=epochs,
        imgsz=img_size,
        device=device,
        project='runs/detect',  # 객체 탐지 결과는 보통 runs/detect에 저장됩니다.
        name=experiment_name
    )
    
    logging.info("Training complete.")
    logging.info(f"Results saved to {results.save_dir}")

    # 3가지 정밀도로 모델 내보내기
    export_model_variants(results, data_yaml_path)


if __name__ == '__main__':
    main()
