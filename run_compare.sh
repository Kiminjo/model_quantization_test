python compare_data_type.py \
    --fp32-engine runs/detect/baseline_yolo11n_imgsz400_20250917_221905/exported_models/FP32/best.engine \
    --fp16-engine runs/detect/baseline_yolo11n_imgsz400_20250917_221905/exported_models/FP16/best.engine \
    --ptq-engine runs/detect/baseline_yolo11n_imgsz400_20250917_221905/exported_models/INT8_PTQ/best.engine \
    --qat-engine runs/qat/qat_best_20250917_224550/best.engine \
    --data data/processed/object_detection/data.yaml \
    --device cuda 


python compare_platform_speed.py \
    --pt-model runs/qat/qat_best_20250917_224550/best.pt \
    --onnx-model runs/qat/qat_best_20250917_224550/best.onnx \
    --trt-model runs/qat/qat_best_20250917_224550/best.engine \
    --device cuda \
    --save-plot results/speed_comparison_by_platform.png