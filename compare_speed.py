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


class ModelSpeedBenchmarker:
    """YOLO 모델 추론 속도 벤치마크 클래스"""
    
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

    def _run_inference_benchmark(self, model, dummy_input, model_name: str) -> Dict:
        """추론 속도 벤치마크 공통 로직"""
        inference_times = []
        
        # 워밍업
        logging.info(f"  💨 {model_name} 워밍업 중...")
        for _ in range(10):
            try:
                _ = model(dummy_input, verbose=False, device=self.device)
            except Exception as e:
                logging.warning(f"  ⚠️  {model_name} 워밍업 중 오류: {e}")
                break
        
        # 실제 측정
        logging.info(f"  ⏱️  {model_name} 추론 속도 측정 중...")
        successful_runs = 0
        for i in range(100):
            try:
                if self.device == 'cuda': torch.cuda.synchronize()
                start_time = time.time()
                _ = model(dummy_input, verbose=False, device=self.device)
                if self.device == 'cuda': torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)
                successful_runs += 1
            except Exception as e:
                if i == 0: logging.warning(f"  ⚠️  {model_name} 추론 중 오류: {e}")
                continue
        
        if successful_runs == 0:
            raise Exception(f"{model_name}의 모든 추론 시도가 실패했습니다.")
        
        avg_inference_time = np.mean(inference_times) * 1000  # ms
        fps = 1.0 / np.mean(inference_times)
        
        logging.info(f"  ✅ {model_name} 완료 - FPS: {fps:.2f} ({successful_runs}/100 성공)")
        return {'avg_inference_time_ms': float(avg_inference_time), 'fps': float(fps), 'successful_runs': successful_runs}

    def benchmark_pt(self, model_path: str) -> Dict:
        """PyTorch 모델(.pt) 벤치마크"""
        logging.info("🔍 PyTorch 모델 벤치마킹 시작...")
        try:
            model = YOLO(model_path).to(self.device)
            logging.info(f"  💻 모델을 {model.device} 디바이스로 이동")
            dummy_input, height, width = self._create_dummy_input(model)
            logging.info(f"  📐 감지된 입력 크기: {height}x{width}")
            
            speed_results = self._run_inference_benchmark(model, dummy_input, "PyTorch")
            return {**{'model_type': 'PyTorch'}, **speed_results}
        except Exception as e:
            logging.error(f"  ❌ PyTorch 모델 벤치마킹 중 오류 발생: {e}", exc_info=True)
            return self._empty_results('PyTorch')

    def benchmark_onnx(self, model_path: str) -> Dict:
        """ONNX 모델(.onnx) 벤치마크"""
        logging.info("🔍 ONNX 모델 벤치마킹 시작...")
        try:
            model = YOLO(model_path, task='detect')
            logging.info(f"  💻 모델 실행 디바이스: {self.device}")
            dummy_input, height, width = self._create_dummy_input(model)
            logging.info(f"  📐 감지된 입력 크기: {height}x{width}")
            
            speed_results = self._run_inference_benchmark(model, dummy_input, "ONNX")
            return {**{'model_type': 'ONNX'}, **speed_results}
        except Exception as e:
            logging.error(f"  ❌ ONNX 모델 벤치마킹 중 오류 발생: {e}", exc_info=True)
            return self._empty_results('ONNX')

    def benchmark_tensorrt(self, model_path: str) -> Dict:
        """TensorRT 모델(.engine) 벤치마크"""
        logging.info("🔍 TensorRT 모델 벤치마킹 시작...")
        try:
            model = YOLO(model_path, task='detect')
            logging.info(f"  💻 모델 실행 디바이스: {self.device}")
            dummy_input, height, width = self._create_dummy_input(model)
            logging.info(f"  📐 감지된 입력 크기: {height}x{width}")

            speed_results = self._run_inference_benchmark(model, dummy_input, "TensorRT")
            return {**{'model_type': 'TensorRT'}, **speed_results}
        except Exception as e:
            logging.error(f"  ❌ TensorRT 모델 벤치마킹 중 오류 발생: {e}", exc_info=True)
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
    
    def _create_dummy_input(self, model) -> Tuple[Image.Image, int, int]:
        """모델에 맞는 더미 입력 생성"""
        height, width = self._get_model_input_size(model)
        dummy_img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), mode='RGB')
        return dummy_img, height, width
    
    def _empty_results(self, model_type: str) -> Dict:
        """빈 결과 반환"""
        return {'model_type': model_type, 'avg_inference_time_ms': 0.0, 'fps': 0.0, 'successful_runs': 0}
    
    def compare_models(self, model_configs: List[Dict]) -> Dict[str, Dict]:
        """여러 모델 성능 비교"""
        results = {}
        for config in model_configs:
            model_path = config['path']
            model_name = config['name']
            model_type = config['type'].lower()
            
            if not os.path.exists(model_path):
                logging.warning(f"⚠️  모델 파일을 찾을 수 없습니다: {model_path}")
                results[model_name] = self._empty_results(model_type.upper())
                continue
            
            if model_type == 'pt':
                results[model_name] = self.benchmark_pt(model_path)
            elif model_type == 'onnx':
                results[model_name] = self.benchmark_onnx(model_path)
            elif model_type in ['tensorrt', 'trt', 'engine']:
                results[model_name] = self.benchmark_tensorrt(model_path)
            else:
                logging.warning(f"⚠️  지원하지 않는 모델 타입: {model_type}")
                results[model_name] = self._empty_results('UNKNOWN')
        
        self.results = results
        return results

    def plot_speed_comparison(self, save_path: str = 'results/speed_comparison.png', show_plot: bool = True):
        """추론 속도 비교 결과 시각화"""
        if not self.results:
            logging.warning("❌ 비교할 결과가 없습니다.")
            return

        model_names = list(self.results.keys())
        inference_times = [self.results[name]['avg_inference_time_ms'] for name in model_names]
        fps_values = [self.results[name]['fps'] for name in model_names]

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle('Model Inference Speed Comparison (INT8 Standard)', fontsize=16, fontweight='bold')
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        # 서브플롯 1: 추론 시간
        ax1 = axes[0]
        bars1 = ax1.bar(model_names, inference_times, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Average Inference Time (Lower is Better)', fontweight='bold')
        ax1.set_ylabel('Time (ms)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        for bar in bars1:
            yval = bar.get_height()
            if yval > 0:
                ax1.text(bar.get_x() + bar.get_width()/2.0, yval + max(inference_times)*0.01, f'{yval:.2f}ms', ha='center', va='bottom', fontweight='bold')

        # 서브플롯 2: FPS
        ax2 = axes[1]
        bars2 = ax2.bar(model_names, fps_values, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_title('Frames Per Second (Higher is Better)', fontweight='bold')
        ax2.set_ylabel('FPS', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        for bar in bars2:
            yval = bar.get_height()
            if yval > 0:
                ax2.text(bar.get_x() + bar.get_width()/2.0, yval + max(fps_values)*0.01, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
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

        summary = "\n" + "="*70 + "\n"
        summary += "🏆 MODEL INFERENCE SPEED COMPARISON SUMMARY\n"
        summary += "="*70 + "\n"
        summary += f"{'Model Name':<15} {'Type':<10} {'Inf Time(ms)':<20} {'FPS':<15} {'Runs':<10}\n"
        summary += "-"*70 + "\n"

        for name, result in self.results.items():
            summary += (f"{name:<15} {result['model_type']:<10} {result['avg_inference_time_ms']:<20.2f} "
                        f"{result['fps']:<15.1f} {result['successful_runs']:<10}\n")

        summary += "-"*70 + "\n"
        valid_results = {k: v for k, v in self.results.items() if v['successful_runs'] > 0}
        if valid_results:
            best_speed = max(valid_results.items(), key=lambda x: x[1]['fps'])
            summary += f"🚀 Best FPS: {best_speed[0]} ({best_speed[1]['fps']:.1f})\n"
        else:
            summary += "⚠️  유효한 결과가 없습니다.\n"
        summary += "="*70
        logging.info(summary)

    def save_results(self, save_path: str = 'results/speed_results.json'):
        """결과를 JSON 파일로 저장"""
        # 저장 경로의 디렉토리가 없으면 생성
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f"💾 벤치마크 결과가 저장되었습니다: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLO Model Inference Speed Comparison (INT8 Standard)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--save-plot', type=str, default='results/speed_comparison.png', help='Plot save path')
    parser.add_argument('--save-results', type=str, default='results/speed_results.json', help='Results save path')
    parser.add_argument('--no-show', action='store_true', help='Do not show plot')
    
    parser.add_argument('--pt-model', type=str, required=True, help='PyTorch model path (.pt)')
    parser.add_argument('--onnx-model', type=str, required=True, help='ONNX model path (.onnx)')
    parser.add_argument('--trt-model', type=str, required=True, help='TensorRT model path (.engine)')
    
    args = parser.parse_args()
    
    model_configs = [
        {'path': args.pt_model, 'name': 'PyTorch', 'type': 'pt'},
        {'path': args.onnx_model, 'name': 'ONNX', 'type': 'onnx'},
        {'path': args.trt_model, 'name': 'TensorRT', 'type': 'tensorrt'},
    ]

    benchmarker = ModelSpeedBenchmarker(device=args.device)
    
    logging.info("🚀 YOLO 모델 추론 속도 비교를 시작합니다...")
    logging.info(f"🖥️  디바이스: {benchmarker.device}")
    
    benchmarker.compare_models(model_configs)
    benchmarker.print_summary_table()
    benchmarker.plot_speed_comparison(save_path=args.save_plot, show_plot=not args.no_show)
    benchmarker.save_results(save_path=args.save_results)
    
    logging.info("\n✅ 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
