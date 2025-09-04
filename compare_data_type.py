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
        """Ultralytics를 사용한 TensorRT 모델 벤치마크 (성능 및 속도)"""
        logging.info(f"🔍 {model_name} 벤치마킹 시작...")
        
        try:
            model = YOLO(model_path, task='detect')
            
            effective_device = 'cuda' if self.device == 'cuda' and torch.cuda.is_available() else 'cpu'
            logging.info(f"  💻 모델 실행 디바이스: {effective_device}")
            
            # 모델 입력 크기 감지
            height, width = self._get_model_input_size(model)
            logging.info(f"  📐 감지된 입력 크기: {height}x{width}")
            
            # 1. Validation 실행 (mAP 계산)
            logging.info("  📊 mAP 계산 중...")
            val_results = model.val(data=self.data_path, verbose=False, imgsz=height, device=self.device)
            map50_95 = val_results.box.map if hasattr(val_results.box, 'map') else 0.0
            map50 = val_results.box.map50 if hasattr(val_results.box, 'map50') else 0.0
            logging.info(f"  ✅ 성능 측정 완료 - mAP50-95: {map50_95:.4f}, mAP50: {map50:.4f}")

            # 2. 추론 속도 측정
            dummy_input, _, _ = self._create_dummy_input(model)
            speed_results = self._run_inference_benchmark(model, dummy_input, model_name)
            
            results = {
                'model_type': 'TensorRT',
                'map50_95': float(map50_95),
                'map50': float(map50),
                'input_size': f"{height}x{width}",
                **speed_results
            }
            
            return results
            
        except Exception as e:
            logging.error(f"  ❌ {model_name} 벤치마킹 중 오류 발생: {e}", exc_info=True)
            return self._empty_results('TensorRT')
            
    def _get_model_input_size(self, model) -> Tuple[int, int]:
        """모델의 입력 크기를 자동으로 감지"""
        try:
            # Ultralytics 모델에서 imgsz 속성 확인
            if hasattr(model, 'overrides') and 'imgsz' in model.overrides:
                imgsz = model.overrides['imgsz']
                if isinstance(imgsz, int): return imgsz, imgsz
                if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2: return imgsz[0], imgsz[1]
            
            # 모델 args에서 imgsz 확인
            if hasattr(model, 'args') and hasattr(model.args, 'imgsz'):
                imgsz = model.args.imgsz
                if isinstance(imgsz, int): return imgsz, imgsz
                if isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2: return imgsz[0], imgsz[1]
            
            # 일반적인 YOLO 입력 크기들을 시도해봄 (작은 크기부터)
            # common_sizes = [320, 384, 400, 416, 480, 512, 640]
            common_sizes = [416]
            
            for size in common_sizes:
                try:
                    # PIL Image로 테스트 입력 생성
                    from PIL import Image
                    import numpy as np
                    test_img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
                    _ = model(test_img, verbose=False)
                    logging.info(f"  📏 감지된 모델 입력 크기: {size}x{size}")
                    return size, size  # 성공하면 해당 크기 반환
                except Exception as e:
                    error_msg = str(e).lower()
                    if "input" in error_msg and "size" in error_msg:
                        # 오류 메시지에서 모델의 최대 크기 추출 시도
                        # 예: "max model size (1, 3, 416, 416)"
                        import re
                        match = re.search(r'max model size.*?(\d+),\s*(\d+)\)', error_msg)
                        if match:
                            max_height, max_width = int(match.group(1)), int(match.group(2))
                            logging.info(f"  📏 오류 메시지에서 추출한 모델 크기: {max_height}x{max_width}")
                            return max_height, max_width
                        continue
                    # 다른 타입의 에러는 해당 크기가 맞다고 가정
                    return size, size
            
            # 모든 크기가 실패하면 기본값 반환 (가장 작은 크기)
            logging.warning("  ⚠️  모델 입력 크기 자동 감지 실패. 기본값 416x416 사용")
            return 416, 416
            
        except Exception:
            # 감지 실패 시 기본값
            logging.warning("  ⚠️  모델 입력 크기 감지 중 오류 발생. 기본값 416x416 사용")
            return 416, 416
            
    def _create_dummy_input(self, model) -> Tuple[Image.Image, int, int]:
        """모델에 맞는 더미 입력 생성"""
        height, width = self._get_model_input_size(model)
        dummy_img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), mode='RGB')
        return dummy_img, height, width

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
            logging.error(f"{model_name}의 모든 추론 시도가 실패했습니다.")
            return {'avg_inference_time_ms': 0.0, 'fps': 0.0, 'successful_runs': 0}
        
        avg_inference_time = np.mean(inference_times) * 1000  # ms
        fps = 1.0 / np.mean(inference_times)
        
        logging.info(f"  ✅ {model_name} 속도 측정 완료 - FPS: {fps:.2f} ({successful_runs}/100 성공)")
        return {'avg_inference_time_ms': float(avg_inference_time), 'fps': float(fps), 'successful_runs': successful_runs}

    def _empty_results(self, model_type: str) -> Dict:
        """빈 결과 반환"""
        return {
            'model_type': model_type,
            'map50_95': 0.0,
            'map50': 0.0,
            'input_size': "0x0",
            'avg_inference_time_ms': 0.0,
            'fps': 0.0,
            'successful_runs': 0,
        }
    
    def compare_models(self, model_configs: List[Dict]) -> Dict[str, Dict]:
        """여러 모델 성능 비교"""
        results = {}
        for i, config in enumerate(model_configs):
            model_path = config['path']
            model_name = config['name']
            
            if not os.path.exists(model_path):
                logging.warning(f"⚠️  모델 파일을 찾을 수 없습니다: {model_path}")
                results[model_name] = self._empty_results('TensorRT')
                continue
            
            results[model_name] = self.benchmark_tensorrt(model_path, model_name)

            # 마지막 모델이 아니면 잠시 대기하여 GPU 안정화
            if i < len(model_configs) - 1:
                wait_time = 10
                logging.info(f"✨ 다음 모델 벤치마킹 전에 {wait_time}초 동안 대기합니다...")
                time.sleep(wait_time)
        
        self.results = results
        return results
    
    def plot_performance_comparison(self, save_path: str = 'results/performance_comparison_by_datatype.png', show_plot: bool = True):
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
        ax.set_title('TensorRT Model Performance Comparison (by Data Type)', fontsize=16, fontweight='bold')
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

    def plot_speed_comparison(self, save_path: str = 'results/speed_comparison_by_datatype.png', show_plot: bool = True):
        """추론 속도 비교 결과 시각화"""
        if not self.results:
            logging.warning("❌ 비교할 결과가 없습니다.")
            return

        model_names = list(self.results.keys())
        inference_times = [self.results[name]['avg_inference_time_ms'] for name in model_names]
        fps_values = [self.results[name]['fps'] for name in model_names]

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        fig.suptitle('TensorRT Model Speed Comparison (by Data Type)', fontsize=16, fontweight='bold')
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

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
        logging.info(f"📊 속도 비교 그래프가 저장되었습니다: {save_path}")

        if show_plot:
            plt.show()

    def print_summary_table(self):
        """결과를 테이블 형태로 출력"""
        if not self.results:
            logging.warning("❌ 출력할 결과가 없습니다.")
            return
        
        summary = "\n" + "="*85 + "\n"
        summary += "🏆 MODEL PERFORMANCE & SPEED COMPARISON SUMMARY\n"
        summary += "="*85 + "\n"
        summary += f"{'Model Name':<15} {'mAP50-95':<15} {'mAP50':<15} {'Inf Time(ms)':<20} {'FPS':<15}\n"
        summary += "-"*85 + "\n"
        
        for name, result in self.results.items():
            summary += (f"{name:<15} {result['map50_95']:<15.4f} {result['map50']:<15.4f} "
                        f"{result['avg_inference_time_ms']:<20.2f} {result['fps']:<15.1f}\n")
        
        summary += "-"*85 + "\n"
        valid_results = {k: v for k, v in self.results.items() if v['successful_runs'] > 0}
        if valid_results:
            best_map = max(valid_results.items(), key=lambda x: x[1]['map50_95'])
            best_speed = max(valid_results.items(), key=lambda x: x[1]['fps'])
            summary += f"🥇 Best mAP50-95: {best_map[0]} ({best_map[1]['map50_95']:.4f})\n"
            summary += f"🚀 Best FPS:      {best_speed[0]} ({best_speed[1]['fps']:.1f})\n"
        else:
            summary += "⚠️  유효한 결과가 없습니다.\n"
        summary += "="*85
        logging.info(summary)
        
    def save_results(self, save_path: str = 'results/fp32_fp16_int8_comparison.json'):
        """결과를 JSON 파일로 저장"""
        # 저장 경로의 디렉토리가 없으면 생성
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f"💾 벤치마크 결과가 저장되었습니다: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLO TensorRT Model Performance and Speed Comparison by Data Type')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Dataset config path')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--save-perf-plot', type=str, default='results/performance_comparison_by_datatype.png', help='Performance plot save path')
    parser.add_argument('--save-speed-plot', type=str, default='results/speed_comparison_by_datatype.png', help='Speed plot save path')
    parser.add_argument('--save-results', type=str, default='results/fp32_fp16_int8_comparison.json', help='Results JSON save path')
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
    benchmarker.plot_performance_comparison(save_path=args.save_perf_plot, show_plot=not args.no_show)
    benchmarker.plot_speed_comparison(save_path=args.save_speed_plot, show_plot=not args.no_show)
    benchmarker.save_results(save_path=args.save_results)
    
    logging.info("\n✅ 모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()
