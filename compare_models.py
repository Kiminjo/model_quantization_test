import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# 필요한 라이브러리 import
try:
    from ultralytics import YOLO
    import torch
    from PIL import Image
except ImportError as e:
    print(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
    print("설치 명령어:")
    print("pip install ultralytics torch matplotlib pillow")


class ModelBenchmarker:
    """YOLO 모델 성능 벤치마크 클래스"""
    
    def __init__(self, data_path: str = 'coco128.yaml', device: str = 'auto'):
        self.data_path = data_path
        self.device = self._setup_device(device)
        self.results = {}
        
    def _setup_device(self, device: str) -> str:
        """디바이스 설정"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def benchmark_ultralytics_pt(self, model_path: str, model_name: str = "PT Model") -> Dict:
        """Ultralytics PT 모델 벤치마크"""
        print(f"\n🔍 {model_name} 벤치마킹 시작...")
        
        try:
            # 모델 로드
            model = YOLO(model_path)
            
            # 모델 입력 크기 자동 감지 및 더미 입력 생성
            print("  🔧 모델 입력 크기 감지 중...")
            dummy_input, height, width = self._create_dummy_input(model)
            print(f"  📐 감지된 입력 크기: {height}x{width}")
            
            # Validation 실행 (mAP 계산)
            print("  📊 mAP 계산 중...")
            val_results = model.val(data=self.data_path, verbose=False, imgsz=height)
            map50_95 = val_results.box.map if hasattr(val_results.box, 'map') else 0.0
            map50 = val_results.box.map50 if hasattr(val_results.box, 'map50') else 0.0
            
            # 추론 속도 측정
            print("  ⏱️  추론 속도 측정 중...")
            inference_times = []
            
            # 워밍업
            for _ in range(10):
                try:
                    _ = model(dummy_input, verbose=False)
                except Exception as e:
                    print(f"  ⚠️  워밍업 중 오류: {e}")
                    break
            
            # 실제 측정
            successful_runs = 0
            for i in range(100):
                try:
                    start_time = time.time()
                    _ = model(dummy_input, verbose=False)
                    end_time = time.time()
                    inference_times.append(end_time - start_time)
                    successful_runs += 1
                except Exception as e:
                    if i == 0:  # 첫 번째 실행에서 실패하면 에러 출력
                        print(f"  ⚠️  추론 중 오류: {e}")
                    continue
            
            if successful_runs == 0:
                raise Exception("모든 추론 시도가 실패했습니다.")
            
            avg_inference_time = np.mean(inference_times) * 1000  # ms
            fps = 1.0 / np.mean(inference_times)
            
            results = {
                'model_type': 'PT',
                'map50_95': float(map50_95),
                'map50': float(map50),
                'avg_inference_time_ms': float(avg_inference_time),
                'fps': float(fps),
                'input_size': f"{height}x{width}",
                'successful_runs': successful_runs
            }
            
            print(f"  ✅ 완료 - mAP50-95: {map50_95:.4f}, FPS: {fps:.2f} ({successful_runs}/100 성공)")
            return results
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            return self._empty_results('PT')
    
    def benchmark_onnx(self, model_path: str, model_name: str = "ONNX Model") -> Dict:
        """Ultralytics를 사용한 ONNX 모델 벤치마크"""
        print(f"\n🔍 {model_name} 벤치마킹 시작...")
        
        try:
            # Ultralytics YOLO로 ONNX 모델 로드
            model = YOLO(model_path, task='detect')
            
            # 모델 입력 크기 자동 감지 및 더미 입력 생성
            print("  🔧 모델 입력 크기 감지 중...")
            dummy_input, height, width = self._create_dummy_input(model)
            print(f"  📐 감지된 입력 크기: {height}x{width}")
            
            # Validation 실행 (mAP 계산)
            print("  📊 mAP 계산 중...")
            val_results = model.val(data=self.data_path, verbose=False, imgsz=height)
            map50_95 = val_results.box.map if hasattr(val_results.box, 'map') else 0.0
            map50 = val_results.box.map50 if hasattr(val_results.box, 'map50') else 0.0
            
            # 추론 속도 측정
            print("  ⏱️  추론 속도 측정 중...")
            inference_times = []
            
            # 워밍업
            for _ in range(10):
                try:
                    _ = model(dummy_input, verbose=False)
                except Exception as e:
                    print(f"  ⚠️  워밍업 중 오류: {e}")
                    break
            
            # 실제 측정
            successful_runs = 0
            for i in range(100):
                try:
                    start_time = time.time()
                    _ = model(dummy_input, verbose=False)
                    end_time = time.time()
                    inference_times.append(end_time - start_time)
                    successful_runs += 1
                except Exception as e:
                    if i == 0:
                        print(f"  ⚠️  추론 중 오류: {e}")
                    continue
            
            if successful_runs == 0:
                raise Exception("모든 추론 시도가 실패했습니다.")
            
            avg_inference_time = np.mean(inference_times) * 1000  # ms
            fps = 1.0 / np.mean(inference_times)
            
            results = {
                'model_type': 'ONNX',
                'map50_95': float(map50_95),
                'map50': float(map50),
                'avg_inference_time_ms': float(avg_inference_time),
                'fps': float(fps),
                'input_size': f"{height}x{width}",
                'successful_runs': successful_runs
            }
            
            print(f"  ✅ 완료 - mAP50-95: {map50_95:.4f}, FPS: {fps:.2f} ({successful_runs}/100 성공)")
            return results
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            return self._empty_results('ONNX')
    
    def benchmark_tensorrt(self, model_path: str, model_name: str = "TensorRT Model") -> Dict:
        """Ultralytics를 사용한 TensorRT 모델 벤치마크"""
        print(f"\n🔍 {model_name} 벤치마킹 시작...")
        
        try:
            # Ultralytics YOLO로 TensorRT 모델 로드
            model = YOLO(model_path, task='detect')
            
            # 모델 입력 크기 자동 감지 및 더미 입력 생성
            print("  🔧 모델 입력 크기 감지 중...")
            dummy_input, height, width = self._create_dummy_input(model)
            print(f"  📐 감지된 입력 크기: {height}x{width}")
            
            # Validation 실행 (mAP 계산)
            print("  📊 mAP 계산 중...")
            val_results = model.val(data=self.data_path, verbose=False, imgsz=height)
            map50_95 = val_results.box.map if hasattr(val_results.box, 'map') else 0.0
            map50 = val_results.box.map50 if hasattr(val_results.box, 'map50') else 0.0
            
            # 추론 속도 측정
            print("  ⏱️  추론 속도 측정 중...")
            inference_times = []
            
            # 워밍업
            for _ in range(10):
                try:
                    _ = model(dummy_input, verbose=False)
                except Exception as e:
                    print(f"  ⚠️  워밍업 중 오류: {e}")
                    break
            
            # 실제 측정
            successful_runs = 0
            for i in range(100):
                try:
                    start_time = time.time()
                    _ = model(dummy_input, verbose=False)
                    end_time = time.time()
                    inference_times.append(end_time - start_time)
                    successful_runs += 1
                except Exception as e:
                    if i == 0:
                        print(f"  ⚠️  추론 중 오류: {e}")
                    continue
            
            if successful_runs == 0:
                raise Exception("모든 추론 시도가 실패했습니다.")
            
            avg_inference_time = np.mean(inference_times) * 1000  # ms
            fps = 1.0 / np.mean(inference_times)
            
            results = {
                'model_type': 'TensorRT',
                'map50_95': float(map50_95),
                'map50': float(map50),
                'avg_inference_time_ms': float(avg_inference_time),
                'fps': float(fps),
                'input_size': f"{height}x{width}",
                'successful_runs': successful_runs
            }
            
            print(f"  ✅ 완료 - mAP50-95: {map50_95:.4f}, FPS: {fps:.2f} ({successful_runs}/100 성공)")
            return results
            
        except Exception as e:
            print(f"  ❌ 오류 발생: {e}")
            return self._empty_results('TensorRT')
    
    def _get_model_input_size(self, model) -> Tuple[int, int]:
        """모델의 입력 크기를 자동으로 감지"""
        try:
            # Ultralytics 모델에서 imgsz 속성 확인
            if hasattr(model, 'overrides') and 'imgsz' in model.overrides:
                imgsz = model.overrides['imgsz']
                if isinstance(imgsz, int):
                    return imgsz, imgsz
                elif isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
                    return imgsz[0], imgsz[1]
            
            # 모델 args에서 imgsz 확인
            if hasattr(model, 'args') and hasattr(model.args, 'imgsz'):
                imgsz = model.args.imgsz
                if isinstance(imgsz, int):
                    return imgsz, imgsz
                elif isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2:
                    return imgsz[0], imgsz[1]
            
            # 일반적인 YOLO 입력 크기들을 시도해봄 (작은 크기부터)
            # common_sizes = [320, 384, 400, 416, 480, 512, 640]
            common_sizes = [416]
            
            for size in common_sizes:
                try:
                    # PIL Image로 테스트 입력 생성
                    test_img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
                    _ = model(test_img, verbose=False)
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
                            print(f"  📏 오류 메시지에서 추출한 모델 크기: {max_height}x{max_width}")
                            return max_height, max_width
                        continue
                    # 다른 타입의 에러는 해당 크기가 맞다고 가정
                    return size, size
            
            # 모든 크기가 실패하면 기본값 반환 (가장 작은 크기)
            return 416, 416
            
        except Exception:
            # 감지 실패 시 기본값
            return 416, 416
    
    def _create_dummy_input(self, model) -> Tuple[np.ndarray, int, int]:
        """모델에 맞는 더미 입력 생성"""
        height, width = self._get_model_input_size(model)
        
        # PIL Image 형태로 더미 입력 생성 (Ultralytics가 선호하는 형식)
        dummy_img = Image.fromarray(
            np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), 
            mode='RGB'
        )
        
        return dummy_img, height, width
    
    def _empty_results(self, model_type: str) -> Dict:
        """빈 결과 반환"""
        return {
            'model_type': model_type,
            'map50_95': 0.0,
            'map50': 0.0,
            'avg_inference_time_ms': 0.0,
            'fps': 0.0,
            'input_size': "0x0",
            'successful_runs': 0
        }
    
    def compare_models(self, model_configs: List[Dict]) -> Dict[str, Dict]:
        """여러 모델 성능 비교"""
        results = {}
        
        for config in model_configs:
            model_path = config['path']
            model_name = config['name']
            model_type = config['type'].lower()
            
            if not os.path.exists(model_path):
                print(f"⚠️  모델 파일을 찾을 수 없습니다: {model_path}")
                results[model_name] = self._empty_results(model_type.upper())
                continue
            
            if model_type == 'pt':
                results[model_name] = self.benchmark_ultralytics_pt(model_path, model_name)
            elif model_type == 'onnx':
                results[model_name] = self.benchmark_onnx(model_path, model_name)
            elif model_type in ['tensorrt', 'trt', 'engine']:
                results[model_name] = self.benchmark_tensorrt(model_path, model_name)
            else:
                print(f"⚠️  지원하지 않는 모델 타입: {model_type}")
                results[model_name] = self._empty_results('UNKNOWN')
        
        self.results = results
        return results
    
    def plot_comparison(self, save_path: str = 'model_comparison.png', show_plot: bool = True):
        """결과 시각화"""
        if not self.results:
            print("❌ 비교할 결과가 없습니다.")
            return
        
        # 데이터 준비
        model_names = list(self.results.keys())
        map50_95_values = [self.results[name]['map50_95'] for name in model_names]
        map50_values = [self.results[name]['map50'] for name in model_names]
        inference_times = [self.results[name]['avg_inference_time_ms'] for name in model_names]
        fps_values = [self.results[name]['fps'] for name in model_names]
        
        # 플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLO Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 색상 설정
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'][:len(model_names)]
        
        # mAP50-95 비교
        ax1 = axes[0, 0]
        bars1 = ax1.bar(model_names, map50_95_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('mAP50-95 Comparison', fontweight='bold', fontsize=12)
        ax1.set_ylabel('mAP50-95', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars1, map50_95_values)):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(map50_95_values)*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # mAP50 비교
        ax2 = axes[0, 1]
        bars2 = ax2.bar(model_names, map50_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('mAP50 Comparison', fontweight='bold', fontsize=12)
        ax2.set_ylabel('mAP50', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars2, map50_values)):
            if value > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(map50_values)*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 추론 시간 비교 (낮을수록 좋음)
        ax3 = axes[1, 0]
        bars3 = ax3.bar(model_names, inference_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('Average Inference Time (Lower is Better)', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Time (ms)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars3, inference_times)):
            if value > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(inference_times)*0.01,
                        f'{value:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        # FPS 비교 (높을수록 좋음)
        ax4 = axes[1, 1]
        bars4 = ax4.bar(model_names, fps_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax4.set_title('FPS Comparison (Higher is Better)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('FPS', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        # 값 표시
        for i, (bar, value) in enumerate(zip(bars4, fps_values)):
            if value > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fps_values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 저장
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 결과 그래프가 저장되었습니다: {save_path}")
        
        if show_plot:
            plt.show()
    
    def print_summary_table(self):
        """결과를 테이블 형태로 출력"""
        if not self.results:
            print("❌ 출력할 결과가 없습니다.")
            return
        
        print("\n" + "="*90)
        print("🏆 MODEL PERFORMANCE COMPARISON SUMMARY")
        print("="*90)
        print(f"{'Model Name':<20} {'Type':<10} {'mAP50-95':<10} {'mAP50':<10} {'Inf Time(ms)':<15} {'FPS':<10} {'Runs':<10}")
        print("-"*90)
        
        for name, result in self.results.items():
            print(f"{name:<20} {result['model_type']:<10} {result['map50_95']:<10.4f} "
                  f"{result['map50']:<10.4f} {result['avg_inference_time_ms']:<15.2f} "
                  f"{result['fps']:<10.1f} {result['successful_runs']:<10}")
        
        print("-"*90)
        
        # 최고 성능 모델 찾기
        valid_results = {k: v for k, v in self.results.items() if v['map50_95'] > 0}
        if valid_results:
            best_map = max(valid_results.items(), key=lambda x: x[1]['map50_95'])
            best_speed = max(valid_results.items(), key=lambda x: x[1]['fps'])
            
            print(f"🥇 Best mAP50-95: {best_map[0]} ({best_map[1]['map50_95']:.4f})")
            print(f"🚀 Best FPS: {best_speed[0]} ({best_speed[1]['fps']:.1f})")
        else:
            print("⚠️  유효한 결과가 없습니다.")
        print("="*90)
    
    def save_results(self, save_path: str = 'benchmark_results.json'):
        """결과를 JSON 파일로 저장"""
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 벤치마크 결과가 저장되었습니다: {save_path}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='YOLO Model Performance Comparison')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Dataset config path')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use')
    parser.add_argument('--save-plot', type=str, default='model_comparison.png', help='Plot save path')
    parser.add_argument('--save-results', type=str, default='benchmark_results.json', help='Results save path')
    parser.add_argument('--no-show', action='store_true', help='Do not show plot')
    
    # 모델 경로 인자들
    parser.add_argument('--pt-model', type=str, help='PyTorch model path (.pt)')
    parser.add_argument('--onnx-model', type=str, help='ONNX model path (.onnx)')
    parser.add_argument('--ptq-trt', type=str, help='PTQ TensorRT model path (.engine)')
    parser.add_argument('--qat-trt', type=str, help='QAT TensorRT model path (.engine)')
    
    args = parser.parse_args()
    
    # 모델 설정 생성
    model_configs = []
    
    if args.pt_model:
        model_configs.append({
            'path': args.pt_model,
            'name': 'PyTorch (.pt)',
            'type': 'pt'
        })
    
    if args.onnx_model:
        model_configs.append({
            'path': args.onnx_model,
            'name': 'ONNX',
            'type': 'onnx'
        })
    
    if args.ptq_trt:
        model_configs.append({
            'path': args.ptq_trt,
            'name': 'PTQ TensorRT',
            'type': 'tensorrt'
        })
    
    if args.qat_trt:
        model_configs.append({
            'path': args.qat_trt,
            'name': 'QAT TensorRT',
            'type': 'tensorrt'
        })
    
    if not model_configs:
        print("❌ 비교할 모델이 지정되지 않았습니다.")
        print("사용법:")
        print("python script.py --pt-model model.pt --onnx-model model.onnx --ptq-trt ptq.engine --qat-trt qat.engine")
        return
    
    # 벤치마크 실행
    benchmarker = ModelBenchmarker(data_path=args.data, device=args.device)
    
    print("🚀 YOLO 모델 성능 비교를 시작합니다...")
    print(f"📊 데이터셋: {args.data}")
    print(f"🖥️  디바이스: {benchmarker.device}")
    print(f"📁 비교할 모델 수: {len(model_configs)}")
    
    # 모델 비교 실행
    results = benchmarker.compare_models(model_configs)
    
    # 결과 출력
    benchmarker.print_summary_table()
    
    # 시각화
    benchmarker.plot_comparison(save_path=args.save_plot, show_plot=not args.no_show)
    
    # 결과 저장
    benchmarker.save_results(save_path=args.save_results)
    
    print("\n✅ 모든 작업이 완료되었습니다!")


if __name__ == "__main__":
    # 대화형 실행을 위한 예시
    if len(os.sys.argv) == 1:
        print("🔧 대화형 모드로 실행 중...")
        
        # 사용자 입력 받기
        model_configs = []
        
        print("\n📝 모델 경로를 입력하세요 (엔터로 건너뛰기):")
        
        pt_path = input("PyTorch 모델 (.pt) 경로: ").strip()
        if pt_path:
            model_configs.append({'path': pt_path, 'name': 'PyTorch (.pt)', 'type': 'pt'})
        
        onnx_path = input("ONNX 모델 (.onnx) 경로: ").strip()
        if onnx_path:
            model_configs.append({'path': onnx_path, 'name': 'ONNX', 'type': 'onnx'})
        
        ptq_path = input("PTQ TensorRT 모델 (.engine) 경로: ").strip()
        if ptq_path:
            model_configs.append({'path': ptq_path, 'name': 'PTQ TensorRT', 'type': 'tensorrt'})
        
        qat_path = input("QAT TensorRT 모델 (.engine) 경로: ").strip()
        if qat_path:
            model_configs.append({'path': qat_path, 'name': 'QAT TensorRT', 'type': 'tensorrt'})
        
        if not model_configs:
            print("❌ 입력된 모델이 없습니다.")
            exit()
        
        data_path = input("데이터셋 경로 (기본값: coco128.yaml): ").strip() or 'coco128.yaml'
        
        # 벤치마크 실행
        benchmarker = ModelBenchmarker(data_path=data_path)
        print(f"\n🚀 벤치마크 시작... (디바이스: {benchmarker.device})")
        
        results = benchmarker.compare_models(model_configs)
        benchmarker.print_summary_table()
        benchmarker.plot_comparison()
        benchmarker.save_results()
    else:
        main()