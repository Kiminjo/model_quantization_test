from pathlib import Path
from ultralytics import YOLO

output_format = 'trt' # 'onnx', 'trt'

# 가장 성능이 좋은 모델('best.pt') 로드
best_model_path = Path('runs/qat/qat_exp_20250823_233519/weights/best.pt')
if best_model_path.exists():
    best_model = YOLO(best_model_path)
    
    if output_format == 'onnx':
        exported_model_path = best_model.export(format='onnx', data='data/processed/object_detection/data.yaml')
    elif output_format == 'trt':
        # TensorRT로 변환
        # int8=True 옵션은 Post-Training Quantization(PTQ)을 적용합니다.
        exported_model_path = best_model.export(format='tensorrt', int8=True, data='data/processed/object_detection/data.yaml')
    
    print(f"Successfully exported model to: {exported_model_path}")
else:
    print(f"Error: Could not find the best model at {best_model_path}")