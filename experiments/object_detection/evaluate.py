"""
Ultralytics YOLOv8 모델 포맷(.pt, .onnx, .engine)별 성능 및 추론 속도를 비교하는 스크립트

- 가장 최근의 학습 결과('runs/detect')를 사용합니다.
- 각 모델 포맷에 대해 'test' 데이터셋으로 평가를 수행합니다.
- mAP50 성능과 추론 시간을 비교하는 그래프를 생성하여 저장합니다.

Author: Injo Kim
Date: 2025-08-21
"""

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import argparse
import re

def find_latest_run_dir(project_path='runs/detect') -> Path | None:
    """ 가장 최근에 실행된 Ultralytics 실험 디렉토리를 찾습니다. """
    project_path = Path(project_path)
    if not project_path.exists():
        return None
    
    run_dirs = [d for d in project_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
        
    latest_run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
    print(f"Found latest run directory: {latest_run_dir}")
    return latest_run_dir

def plot_results(df: pd.DataFrame, save_dir: Path):
    """ 평가 결과를 받아 막대그래프로 시각화하고 저장합니다. """
    
    # 스타일 설정
    sns.set_theme(style="whitegrid")

    # 1. mAP50 성능 비교 그래프
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(x=df.index, y='mAP50', data=df, palette='viridis')
    ax1.set_title('Model Performance Comparison (mAP@.50)', fontsize=16)
    ax1.set_xlabel('Model Format', fontsize=12)
    ax1.set_ylabel('mAP@.50', fontsize=12)
    
    # 막대 위에 값 표시
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                     textcoords='offset points')
    
    plt.ylim(0, max(df['mAP50']) * 1.2)
    performance_plot_path = save_dir / 'performance_comparison.png'
    plt.savefig(performance_plot_path)
    print(f"Performance comparison plot saved to: {performance_plot_path}")
    plt.close()

    # 2. 추론 시간 비교 그래프
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x=df.index, y='Inference Time (ms)', data=df, palette='plasma')
    ax2.set_title('Inference Time Comparison per Image', fontsize=16)
    ax2.set_xlabel('Model Format', fontsize=12)
    ax2.set_ylabel('Time (ms)', fontsize=12)

    # 막대 위에 값 표시
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.2f} ms', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                     textcoords='offset points')

    plt.ylim(0, max(df['Inference Time (ms)']) * 1.2)
    speed_plot_path = save_dir / 'speed_comparison.png'
    plt.savefig(speed_plot_path)
    print(f"Speed comparison plot saved to: {speed_plot_path}")
    plt.close()

    # 3. FPS 비교 그래프
    plt.figure(figsize=(10, 6))
    ax3 = sns.barplot(x=df.index, y='FPS', data=df, palette='cubehelix')
    ax3.set_title('FPS Comparison', fontsize=16)
    ax3.set_xlabel('Model Format', fontsize=12)
    ax3.set_ylabel('FPS (Frames Per Second)', fontsize=12)

    # 막대 위에 값 표시
    for p in ax3.patches:
        ax3.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                     textcoords='offset points')
    
    plt.ylim(0, max(df['FPS']) * 1.2)
    fps_plot_path = save_dir / 'fps_comparison.png'
    plt.savefig(fps_plot_path)
    print(f"FPS comparison plot saved to: {fps_plot_path}")
    plt.close()

def get_imgsz_from_dir(model_dir: Path) -> int:
    """ experiment directory name에서 'imgsz' 값을 파싱합니다. """
    match = re.search(r'imgsz(\d+)', model_dir.name)
    if match:
        img_size = int(match.group(1))
        print(f"Parsed image size from directory name: {img_size}")
        return img_size
    
    default_img_size = 400  # train.py's default
    print(f"Warning: Could not parse image size from directory name. Using default: {default_img_size}")
    return default_img_size

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare YOLOv8 model formats (.pt, .onnx, .engine).")
    parser.add_argument('--model_dir', type=str, default=None,
                        help="Path to the specific model run directory (e.g., 'runs/detect/exp'). "
                             "If not provided, the latest run will be automatically used.")
    args = parser.parse_args()
    
    # --- 1. 경로 설정 및 모델 디렉토리 결정 ---
    project_root = Path(__file__).resolve().parent.parent.parent
    
    if args.model_dir:
        model_dir = Path(args.model_dir)
        print(f"Using specified model directory: {model_dir}")
    else:
        print("No model directory specified, finding the latest run...")
        model_dir = find_latest_run_dir(project_root / 'runs' / 'detect')

    if not model_dir or not model_dir.exists():
        if args.model_dir:
            print(f"Error: Specified model directory not found at '{model_dir}'")
        else:
            print("Error: No training runs found in 'runs/detect'. Please run train.py first.")
        sys.exit(1)

    img_size = get_imgsz_from_dir(model_dir)

    weights_dir = model_dir / 'weights'
    pt_model_path = weights_dir / 'best.pt'
    if not pt_model_path.exists():
        print(f"Error: 'best.pt' not found in {weights_dir}")
        sys.exit(1)

    # --- 2. 다른 포맷으로 모델 변환 (필요시) ---
    onnx_model_path = pt_model_path.with_suffix('.onnx')
    tensorrt_model_path = pt_model_path.with_suffix('.engine')
    
    base_model = YOLO(pt_model_path)
    
    if not onnx_model_path.exists():
        print(f"ONNX model not found. Exporting from {pt_model_path} with imgsz={img_size}...")
        base_model.export(format='onnx', imgsz=img_size)
    
    if not tensorrt_model_path.exists():
        print(f"TensorRT model not found. Exporting from {pt_model_path} with imgsz={img_size}...")
        base_model.export(format='tensorrt', int8=True, imgsz=img_size)

    # --- 3. 평가 데이터셋 경로 설정 ---
    data_yaml_path = project_root / 'data' / 'processed' / 'object_detection' / 'data.yaml'
    if not data_yaml_path.exists():
        print(f"Error: data.yaml not found at '{data_yaml_path}'")
        sys.exit(1)

    # --- 4. 각 모델 포맷별 평가 수행 ---
    model_paths = {
        'PyTorch (.pt)': pt_model_path,
        'ONNX (.onnx)': onnx_model_path,
        'TensorRT (.engine)': tensorrt_model_path,
    }
    
    results_data = {}

    for name, path in model_paths.items():
        print(f"\n--- Evaluating {name} model ---")
        model = YOLO(path)
        metrics = model.val(data=str(data_yaml_path), split='test', imgsz=img_size)
        
        results_data[name] = {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'Inference Time (ms)': metrics.speed['inference'],
        }

    # --- 5. 결과 정리 및 출력 ---
    results_df = pd.DataFrame(results_data).T
    # FPS 계산 추가
    results_df['FPS'] = 1000 / results_df['Inference Time (ms)']
    
    print("\n--- Overall Evaluation Results ---")
    print(results_df)

    # --- 6. 결과 시각화 및 저장 ---
    plot_results(results_df, model_dir)
    print("\nEvaluation complete.")

if __name__ == '__main__':
    main()
