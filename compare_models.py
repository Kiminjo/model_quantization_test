# -*- coding: utf-8 -*-
"""
이 스크립트는 기존 모델(Baseline)과 최적화된 신규 모델(Optimized)의 성능 및 속도를 비교하기 위해 작성되었습니다.
주요 목적은 다음과 같습니다.
1. ONNX FP32 모델을 베이스라인으로 설정합니다.
2. TensorRT INT8 QAT 모델을 최적화된 모델로 설정합니다.
3. 두 모델의 mAP(성능)와 FPS(속도)를 각각 측정합니다.
4. 성능 및 속도 개선율을 계산하고 결과를 요약하여 출력합니다.
"""

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


class ModelComparator:
    """YOLO 모델 성능 및 속도 비교 클래스"""
    
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
        if self.device != 'cuda':
            logging.warning("TensorRT 모델은 CUDA에서만 실행 가능합니다. `device`를 'cuda'로 설정합니다.")
            return 'cuda'
        return device
    
    def benchmark_model(self, model_path: str, model_name: str, model_type: str) -> Optional[Dict]:
        """단일 모델의 성능(mAP)과 속도(FPS)를 벤치마킹"""
        logging.info(f"🔍 '{model_name}' ({model_type}) 모델 벤치마킹 시작...")
        
        if not os.path.exists(model_path):
            logging.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            return None

        try:
            model = YOLO(model_path, task='detect')
            
            # --- 성능 측정 (mAP) ---
            logging.info(f"  📊 [{model_name}] mAP 계산 중...")
            height, width = self._get_model_input_size(model)
            val_results = model.val(data=self.data_path, verbose=False, imgsz=height, device=self.device)
            map50_95 = float(val_results.box.map) if hasattr(val_results.box, 'map') else 0.0
            map50 = float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0.0
            logging.info(f"  ✅ [{model_name}] mAP50-95: {map50_95:.4f}, mAP50: {map50:.4f}")

            # --- 속도 측정 (FPS) ---
            logging.info(f"  ⏱️  [{model_name}] 추론 속도 측정 중...")
            dummy_input, _, _ = self._create_dummy_input(model, height, width)
            
            # 워밍업
            for _ in range(10):
                _ = model(dummy_input, verbose=False, device=self.device)

            # 실제 측정
            inference_times = []
            for _ in range(100):
                torch.cuda.synchronize()
                start_time = time.time()
                _ = model(dummy_input, verbose=False, device=self.device)
                torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)
            
            avg_inference_time = np.mean(inference_times) * 1000  # ms
            fps = 1.0 / np.mean(inference_times)
            logging.info(f"  ✅ [{model_name}] 평균 추론 시간: {avg_inference_time:.2f}ms, FPS: {fps:.2f}")

            return {
                'model_name': model_name,
                'model_type': model_type,
                'map50_95': map50_95,
                'map50': map50,
                'avg_inference_time_ms': avg_inference_time,
                'fps': fps,
            }

        except Exception as e:
            logging.error(f"❌ '{model_name}' 벤치마킹 중 오류 발생: {e}", exc_info=True)
            return None
            
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
    
    def _create_dummy_input(self, model, height, width) -> Tuple[Image.Image, int, int]:
        """모델에 맞는 더미 입력 생성"""
        dummy_img = Image.fromarray(np.random.randint(0, 255, (height, width, 3), dtype=np.uint8), mode='RGB')
        return dummy_img, height, width
    
    def compare_and_summarize(self, baseline_path: str, optimized_path: str):
        """두 모델을 비교하고 결과를 요약"""
        baseline_results = self.benchmark_model(baseline_path, "ONNX FP32 (Baseline)", "ONNX")
        optimized_results = self.benchmark_model(optimized_path, "TensorRT INT8-QAT (Optimized)", "TensorRT")

        if not baseline_results or not optimized_results:
            logging.error("❌ 벤치마크 실패로 비교를 진행할 수 없습니다.")
            return

        # 개선율 계산
        map_improvement = 0
        if baseline_results['map50_95'] > 0:
            map_improvement = ((optimized_results['map50_95'] - baseline_results['map50_95']) / baseline_results['map50_95']) * 100
        
        speed_improvement = 0
        if baseline_results['fps'] > 0:
            speed_improvement = ((optimized_results['fps'] - baseline_results['fps']) / baseline_results['fps']) * 100
            
        self.results = {
            "baseline": baseline_results,
            "optimized": optimized_results,
            "improvements": {
                "map50_95_improvement_percent": map_improvement,
                "speed_improvement_percent_fps": speed_improvement
            }
        }
        
        self.print_summary()

    def print_summary(self):
        """결과 요약 출력"""
        if not self.results:
            logging.warning("❌ 출력할 요약 정보가 없습니다.")
            return
            
        base = self.results['baseline']
        opt = self.results['optimized']
        imp = self.results['improvements']

        summary = "\n" + "="*80 + "\n"
        summary += "🏆 MODEL COMPARISON SUMMARY: ONNX FP32 vs TensorRT INT8-QAT\n"
        summary += "="*80 + "\n"
        summary += f"{'Metric':<25} {'ONNX FP32 (Baseline)':<25} {'TensorRT INT8-QAT (Optimized)':<30}\n"
        summary += "-"*80 + "\n"
        summary += f"{'mAP50-95':<25} {base['map50_95']:<25.4f} {opt['map50_95']:<30.4f}\n"
        summary += f"{'mAP50':<25} {base['map50']:<25.4f} {opt['map50']:<30.4f}\n"
        summary += f"{'Inference Time (ms)':<25} {base['avg_inference_time_ms']:<25.2f} {opt['avg_inference_time_ms']:<30.2f}\n"
        summary += f"{'FPS':<25} {base['fps']:<25.2f} {opt['fps']:<30.2f}\n"
        summary += "="*80 + "\n"
        summary += "📊 IMPROVEMENT ANALYSIS\n"
        summary += "-"*80 + "\n"
        summary += f"  - Performance (mAP50-95): {imp['map50_95_improvement_percent']:+.2f} % "
        summary += "(Positive is better, Negative is worse)\n"
        summary += f"  - Speed (FPS): {imp['speed_improvement_percent_fps']:+.2f} % "
        summary += "(Positive is faster, Negative is slower)\n"
        summary += "="*80
        logging.info(summary)

    def save_results(self, save_path: str = 'results/comparison_results.json'):
        """결과를 JSON 파일로 저장"""
        if not self.results:
            logging.warning("❌ 저장할 결과가 없습니다.")
            return
        
        # 저장 경로의 디렉토리가 없으면 생성
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logging.info(f"💾 비교 결과가 저장되었습니다: {save_path}")

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description='Compare ONNX FP32 and TensorRT INT8-QAT models.')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Dataset config path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda'], help='Device to use (TensorRT requires CUDA)')
    parser.add_argument('--onnx-model', type=str, required=True, help='ONNX FP32 model path (baseline)')
    parser.add_argument('--qat-trt-model', type=str, required=True, help='QAT TensorRT model path (optimized)')
    parser.add_argument('--save-results', type=str, default='results/comparison_results.json', help='Results save path')
    
    args = parser.parse_args()
    
    comparator = ModelComparator(data_path=args.data, device=args.device)
    
    logging.info("🚀 모델 비교를 시작합니다...")
    comparator.compare_and_summarize(args.onnx_model, args.qat_trt_model)
    comparator.save_results(save_path=args.save_results)
    logging.info("\n✅ 모든 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()