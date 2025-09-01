import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 필요한 라이브러리 import
try:
    from ultralytics import YOLO
    import torch
    from PIL import Image
except ImportError as e:
    logging.error(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
    logging.info("설치 명령어: pip install ultralytics torch matplotlib pillow")
    exit()


class ModelPerformanceBenchmarker:
    """YOLO 모델 성능 벤치마크 클래스 (mAP 비교 중심)"""
    
    def __init__(self, data_path: str = 'coco128.yaml', device: str = 'auto'):
        self.data_path = data_path
        self.device = self._setup_device(device)
        self.results = {}
        
    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device in ['auto', 'cuda']:
            if torch.cuda.is_available():
                logging.info(f"✅ CUDA 사용 가능. GPU: {torch.cuda.get_device_name(0)}")
                return 'cuda'
            else:
                logging.warning("⚠️  CUDA를 사용할 수 없습니다. CPU로 실행됩니다.")
                return 'cpu'
        return device

    def benchmark_tensorrt(self, model_path: str, model_name: str) -> Dict:
        """Ultralytics를 사용한 TensorRT 모델 벤치마크"""
        logging.info(f"🔍 {model_name} 벤치마킹 시작...")
        
        try:
            model = YOLO(model_path, task='detect')
            
            effective_device = 'cuda' if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            logging.info(f"  💻 모델 실행 디바이스: {effective_device}")
            
            # 모델 입력 크기 감지
            height, width = self._get_model_input_size(model)
            logging.info(f"  📐 감지된 입력 크기: {height}x{width}")
            
            # Validation 실행 (mAP 계산)
            logging.info("  📊 mAP 계산 중...")
            val_results = model.val(data=self.data_path, verbose=False, imgsz=height, device=self.device)
            map50_95 = val_results.box.map if hasattr(val_results.box, 'map') else 0.0
            map50 = val_results.box.map50 if hasattr(val_results.box, 'map50') else 0.0
            
            results = {
                'model_type': 'TensorRT',
                'map50_95': float(map50_95),
                'map50': float(map50),
                'input_size': f"{height}x{width}",
            }
            
            logging.info(f"  ✅ 완료 - mAP50-95: {map50_95:.4f}, mAP50: {map50:.4f}")
            return results
            
        except Exception as e:
            logging.error(f"  ❌ {model_name} 벤치마킹 중 오류 발생: {e}", exc_info=True)
            return self._empty_results('TensorRT')
            
    def _get_model_input_size(self, model) -> Tuple[int, int]:
        """모델의 입력 크기를 자동으로 감지"""
        try:
            if hasattr(model, 'overrides') and 'imgsz' in model.overrides:
                imgsz = model.overrides['imgsz']
                if isinstance(imgsz, int): return imgsz, imgsz
                if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2: return imgsz[0], imgsz[1]
            if hasattr(model, 'args') and hasattr(model.args, 'imgsz'):
                imgsz = model.args.imgsz
                if isinstance(imgsz, int): return imgsz, imgsz
                if isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2: return imgsz[0], imgsz[1]
            return 640, 640
        except Exception:
            return 640, 640
            
    def _empty_results(self, model_type: str) -> Dict:
        """빈 결과 반환"""
        return {
            'model_type': model_type,
            'map50_95': 0.0,
            'map50': 0.0,
            'input_size': "0x0",
        }
    
    def compare_models(self, model_configs: List[Dict]) -> Dict[str, Dict]:
        """여러 모델 성능 비교"""
        results = {}
        for config in model_configs:
            model_path = config['path']
            model_name = config['name']
            
            if not os.path.exists(model_path):
                logging.warning(f"⚠️  모델 파일을 찾을 수 없습니다: {model_path}")
                results[model_name] = self._empty_results('TensorRT')
                continue
            
            results[model_name] = self.benchmark_tensorrt(model_path, model_name)
        
        self.results = results
        return results
    
    def plot_performance_comparison(self, save_path: str = 'results/performance_comparison.png', show_plot: bool = True):
        """성능 비교 결과 시각화"""
        if not self.results:
            logging.warning("❌ 비교할 결과가 없습니다.")
            return
        
        model_names = list(self.results.keys())
        map50_95_values = [self.results[name]['map50_95'] for name in model_names]
        map50_values = [self.results[name]['map50'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 8))
        rects1 = ax.bar(x - width/2, map50_95_values, width, label='mAP50-95', color='#4ECDC4', alpha=0.8, edgecolor='black')
        rects2 = ax.bar(x + width/2, map50_values, width, label='mAP50', color='#FF6B6B', alpha=0.8, edgecolor='black')
        
        ax.set_ylabel('mAP Score', fontweight='bold')
        ax.set_title('TensorRT Model Performance Comparison (mAP)', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:
                    ax.annotate(f'{height:.4f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontweight='bold')

        autolabel(rects1)
        autolabel(rects2)
        
        fig.tight_layout()
        
        # 저장 경로의 디렉토리가 없으면 생성
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"📊 결과 그래프가 저장되었습니다: {save_path}")
        
        if show_plot:
            plt.show()

    def print_summary_table(self):
        """결과를 테이블 형태로 출력"""
        if not self.results:
            logging.warning("❌ 출력할 결과가 없습니다.")
            return
        
        summary = "\n" + "="*60 + "\n"
        summary += "🏆 MODEL PERFORMANCE COMPARISON SUMMARY\n"
        summary += "="*60 + "\n"
        summary += f"{'Model Name':<15} {'Type':<10} {'mAP50-95':<15} {'mAP50':<15}\n"
        summary += "-"*60 + "\n"
        
        for name, result in self.results.items():
            summary += f"{name:<15} {result['model_type']:<10} {result['map50_95']:<15.4f} {result['map50']:<15.4f}\n"
        
        summary += "-"*60 + "\n"
        valid_results = {k: v for k, v in self.results.items() if v['map50_95'] > 0}
        if valid_results:
            best_map = max(valid_results.items(), key=lambda x: x[1]['map50_95'])
            summary += f"🥇 Best mAP50-95: {best_map[0]} ({best_map[1]['map50_95']:.4f})\n"
        else:
            summary += "⚠️  유효한 결과가 없습니다.\n"
        summary += "="*60
        logging.info(summary)
        
    def save_results(self, save_path: str = 'results/performance_results.json'):
        """결과를 JSON 파일로 저장"""
        # 저장 경로의 디렉토리가 없으면 생성
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f"💾 벤치마크 결과가 저장되었습니다: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLO TensorRT Model Performance Comparison')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Dataset config path')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--save-plot', type=str, default='results/performance_comparison.png', help='Plot save path')
    parser.add_argument('--save-results', type=str, default='results/performance_results.json', help='Results save path')
    parser.add_argument('--no-show', action='store_true', help='Do not show plot')
    
    parser.add_argument('--fp32-engine', type=str, help='FP32 TensorRT model path (.engine)')
    parser.add_argument('--fp16-engine', type=str, help='FP16 TensorRT model path (.engine)')
    parser.add_argument('--ptq-engine', type=str, help='INT8 PTQ TensorRT model path (.engine)')
    parser.add_argument('--qat-engine', type=str, help='INT8 QAT TensorRT model path (.engine)')
    
    args = parser.parse_args()
    
    model_configs = []
    if args.fp32_engine:
        model_configs.append({'path': args.fp32_engine, 'name': 'FP32'})
    if args.fp16_engine:
        model_configs.append({'path': args.fp16_engine, 'name': 'FP16'})
    if args.ptq_engine:
        model_configs.append({'path': args.ptq_engine, 'name': 'INT8_PTQ'})
    if args.qat_engine:
        model_configs.append({'path': args.qat_engine, 'name': 'INT8_QAT'})
    
    if not model_configs:
        logging.error("❌ 비교할 모델이 지정되지 않았습니다. --fp32-engine, --fp16-engine, --ptq-engine, --qat-engine 중 하나 이상을 지정해주세요.")
        return

    benchmarker = ModelPerformanceBenchmarker(data_path=args.data, device=args.device)
    
    logging.info("🚀 YOLO 모델 성능 비교를 시작합니다...")
    logging.info(f"📊 데이터셋: {args.data}")
    logging.info(f"🖥️  디바이스: {benchmarker.device}")
    
    benchmarker.compare_models(model_configs)
    benchmarker.print_summary_table()
    benchmarker.plot_performance_comparison(save_path=args.save_plot, show_plot=not args.no_show)
    benchmarker.save_results(save_path=args.save_results)
    
    logging.info("\n✅ 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
