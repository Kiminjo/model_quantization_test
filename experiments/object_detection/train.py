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
    export_params: dict,
    imgsz: int
):
    """지정된 정밀도로 모델을 준비하고 내보냅니다."""
    logging.info(f"Preparing and exporting {precision} model...")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        model = YOLO(source_pt_path)
        
        # export_params에 imgsz 추가
        export_params['imgsz'] = imgsz
        
        exported_path = model.export(format="engine", **export_params)
        
        # 내보낸 모델을 타겟 디렉토리로 이동
        exported_file = Path(exported_path)
        destination_path = target_dir / exported_file.name
        shutil.move(str(exported_file), str(destination_path))
        
        logging.info(f"Successfully exported {precision} model to: {destination_path}")
    except Exception as e:
        logging.error(f"Failed to export {precision} model: {e}", exc_info=True)

def export_model_variants(results, data_yaml_path):
    """FP32, FP16, INT8 모델들을 내보냅니다."""
    weights_dir = Path(results.save_dir) / 'weights'
    best_pt_path = weights_dir / 'best.pt'
    
    if not best_pt_path.exists():
        logging.error(f"Best weights not found at {best_pt_path}")
        return

    # 모델 저장 디렉토리 설정 - results.save_dir의 상위 디렉토리를 사용
    # results.save_dir는 'runs/detect/experiment_name' 형태
    experiment_dir = Path(results.save_dir)
    export_base_dir = experiment_dir / 'exported_models'
    fp32_dir = export_base_dir / 'FP32'
    fp16_dir = export_base_dir / 'FP16'
    int8_dir = export_base_dir / 'INT8_PTQ'
    
    # 이미지 사이즈 설정 (에러 해결을 위해 추가)
    imgsz = 400

    # FP32 모델 export
    prepare_and_export_quantized_model(
        precision='FP32',
        source_pt_path=best_pt_path,
        target_dir=fp32_dir,
        export_params={},
        imgsz=imgsz
    )

    # FP16 모델 export
    prepare_and_export_quantized_model(
        precision='FP16',
        source_pt_path=best_pt_path,
        target_dir=fp16_dir,
        export_params={'half': True},
        imgsz=imgsz
    )

    # INT8-PTQ 모델 export
    prepare_and_export_quantized_model(
        precision='INT8',
        source_pt_path=best_pt_path,
        target_dir=int8_dir,
        export_params={'int8': True, 'data': str(data_yaml_path)},
        imgsz=imgsz
    )
    
    logging.info(f"All models exported to: {export_base_dir}")
    
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

    # 사전 학습된 YOLOv11n 모델 로드
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
