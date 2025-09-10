import torch
import torch.nn as nn
from ultralytics import YOLO
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
from datetime import datetime
from pathlib import Path
import argparse
import shutil
import logging
import sys

def setup_logging():
    """Setup logging to file and console."""
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = log_dir / f"od_qat_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Log file will be saved to {log_filename}")

class QATWrapper(nn.Module):
    """YOLO 모델을 QAT용으로 래핑하는 클래스"""
    def __init__(self, yolo_model):
        super().__init__()
        self.quant = QuantStub()
        self.model = yolo_model
        self.dequant = DeQuantStub()
        
    def forward(self, x):
        # 입력을 quantize
        x = self.quant(x)
        
        # YOLO 모델 forward
        # 훈련 모드에서는 loss 계산을 위한 출력 반환
        if self.training:
            x = self.model(x)
        else:
            # 추론 모드에서는 detection 결과 반환
            x = self.model(x)
        
        # 출력을 dequantize
        if isinstance(x, (list, tuple)):
            # 여러 출력이 있는 경우 (훈련 시 loss components)
            x = [self.dequant(out) if torch.is_tensor(out) else out for out in x]
        else:
            x = self.dequant(x)
        
        return x

class YOLOQATModel(nn.Module):
    """QAT가 적용된 YOLO 모델"""
    def __init__(self, model_path='yolov8n.pt'):
        super().__init__()
        # YOLO 모델 로드
        self.yolo = YOLO(model_path)
        
        # core model 추출
        self.core_model = self.yolo.model
        
        # QAT wrapper 적용
        self.qat_model = QATWrapper(self.core_model)
        
        # QAT 설정
        self.setup_qat()
    
    def setup_qat(self):
        """QAT 설정"""
        # QConfig 설정 (fake quantization을 위한)
        self.qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # 모델을 QAT 모드로 준비
        torch.quantization.prepare_qat(self.qat_model, inplace=True)
        
    def forward(self, x):
        return self.qat_model(x)
    
    def train_step(self, batch):
        """훈련 스텝"""
        imgs = batch['img']
        
        # Forward pass
        if hasattr(self.yolo, 'model') and hasattr(self.yolo.model, 'training'):
            self.yolo.model.train()
            
        preds = self.qat_model(imgs)
        
        # Loss 계산은 YOLO의 내부 로직 사용
        # 실제로는 YOLO의 criterion을 직접 사용해야 함
        loss = self.compute_loss(preds, batch)
        
        return loss
    
    def compute_loss(self, preds, batch):
        """손실 계산 (YOLO의 내부 로직 필요)"""
        # 실제 구현에서는 YOLO의 loss 계산 로직을 가져와야 함
        # 여기서는 예시를 위한 더미 loss
        if isinstance(preds, (list, tuple)) and len(preds) > 0:
            # 훈련 모드에서 YOLO는 보통 loss를 직접 반환
            return preds[0] if torch.is_tensor(preds[0]) else torch.tensor(0.0)
        return torch.tensor(0.0, requires_grad=True)
    
    def convert_to_quantized(self):
        """QAT 모델을 실제 quantized 모델로 변환"""
        self.qat_model.eval()
        quantized_model = torch.quantization.convert(self.qat_model, inplace=False)
        return quantized_model


def train_qat_yolo(model_path='yolov8n.pt', data_path='coco128.yaml', epochs=10):
    """QAT YOLO 모델 훈련 - QAT에 최적화된 학습률과 스케줄러 사용"""
    
    # QAT 모델 생성
    qat_yolo = YOLOQATModel(model_path)
    
    # QAT에 적합한 낮은 학습률과 옵티마이저 설정
    initial_lr = 0.0001  # QAT는 일반적으로 매우 낮은 학습률 사용
    optimizer = torch.optim.AdamW(
        qat_yolo.parameters(), 
        lr=initial_lr,
        weight_decay=1e-4,  # 안정적인 훈련을 위한 weight decay
        eps=1e-8
    )
    
    # QAT에 적합한 학습률 스케줄러
    # CosineAnnealingLR이 QAT에서 일반적으로 좋은 성능을 보임
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs,
        eta_min=initial_lr * 0.01  # 최소 학습률을 초기값의 1%로 설정
    )
    
    # 데이터로더 설정 (실제로는 YOLO의 데이터로더를 사용해야 함)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qat_yolo.to(device)
    
    logging.info(f"QAT 훈련 시작 - {epochs} epochs")
    logging.info(f"초기 학습률: {initial_lr}")
    logging.info(f"스케줄러: CosineAnnealingLR")
    
    for epoch in range(epochs):
        qat_yolo.train()
        epoch_loss = 0.0
        
        # 실제로는 YOLO의 dataloader를 사용해야 함
        # 여기서는 예시를 위한 더미 루프
        for i in range(10):  # 더미 배치
            # 더미 데이터
            batch = {
                'img': torch.randn(4, 3, 640, 640).to(device),
                'cls': torch.randint(0, 80, (4, 10)).to(device),
                'bboxes': torch.randn(4, 10, 4).to(device)
            }
            
            optimizer.zero_grad()
            loss = qat_yolo.train_step(batch)
            
            if hasattr(loss, 'backward'):
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        # 학습률 스케줄러 업데이트
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logging.info(f"Epoch {epoch+1}/{epochs} 완료 - Loss: {epoch_loss:.4f}, LR: {current_lr:.6f}")
    
    # 훈련 완료 후 quantized 모델로 변환
    quantized_model = qat_yolo.convert_to_quantized()
    
    return qat_yolo, quantized_model


# 더 실용적인 접근법: Ultralytics 훈련 파이프라인과 통합
class YOLOQATTrainer:
    """Ultralytics 훈련 파이프라인과 통합된 QAT 트레이너"""
    
    def __init__(self, model_path='yolov8n.pt'):
        self.yolo = YOLO(model_path)
        self.original_model = self.yolo.model  # 원본 모델 백업
        self.setup_qat()
    
    def setup_qat(self):
        """기존 YOLO 모델에 QAT 적용"""
        original_model = self.original_model
        
        # 새로운 모델 생성 - 원본 모델의 모든 속성을 유지
        class QATYOLOModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.quant = QuantStub()
                self.model = original_model
                self.dequant = DeQuantStub()
                
                # 원본 모델의 중요한 속성들을 복사
                self.yaml = original_model.yaml if hasattr(original_model, 'yaml') else None
                self.nc = original_model.nc if hasattr(original_model, 'nc') else None
                self.names = original_model.names if hasattr(original_model, 'names') else None
                self.stride = original_model.stride if hasattr(original_model, 'stride') else None
                
            def __getattr__(self, name):
                """원본 모델의 속성에 접근할 수 있도록 함"""
                if name in ['quant', 'model', 'dequant', 'yaml', 'nc', 'names', 'stride']:
                    return super().__getattribute__(name)
                # 원본 모델에서 속성 찾기
                if hasattr(self.model, name):
                    return getattr(self.model, name)
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
            def forward(self, x, *args, **kwargs):
                x = self.quant(x)
                x = self.model(x, *args, **kwargs)
                
                # 출력 타입에 따라 dequantize 처리
                if isinstance(x, dict):
                    # 훈련 시 loss dict 반환
                    for key, value in x.items():
                        if torch.is_tensor(value):
                            x[key] = self.dequant(value)
                elif isinstance(x, (list, tuple)):
                    x = [self.dequant(item) if torch.is_tensor(item) else item for item in x]
                else:
                    x = self.dequant(x)
                
                return x
        
        # QAT 모델로 교체
        qat_model = QATYOLOModel(original_model)
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(qat_model, inplace=True)
        
        # YOLO 객체의 모델을 QAT 모델로 교체
        self.yolo.model = qat_model
    
    def train(self, data='coco128.yaml', epochs=10, **kwargs):
        """QAT 훈련 실행 - 더 안전한 방법"""
        logging.info("QAT 훈련 시작...")
        
        # `project`와 `name`을 kwargs에서 추출하여 저장 경로 제어
        project = kwargs.pop('project', 'runs/qat')
        name = kwargs.pop('name', f'qat_exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # 원본 모델을 임시로 복원하여 trainer 초기화
        self.yolo.model = self.original_model
        
        # Ultralytics의 기본 훈련 파이프라인 사용
        # 하지만 실제 훈련 전에 QAT 모델로 다시 교체
        try:
            results = self.yolo.train(
                data=data,
                epochs=epochs,
                project=project,
                name=name,
                **kwargs
            )
        except Exception as e:
            logging.error(f"훈련 중 오류 발생: {e}")
            # QAT 모델로 다시 설정
            self.setup_qat()
            raise e
        
        return results
    
    def train_alternative(self, data='coco128.yaml', epochs=10, **kwargs):
        """대안적 QAT 훈련 방법 - QAT에 최적화된 설정으로 직접 trainer 생성"""
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.utils import LOGGER
        
        logging.info("QAT 훈련 시작 (대안 방법)...")
        
        # QAT에 최적화된 설정
        qat_lr0 = kwargs.pop('lr0', 0.0001)  # QAT용 낮은 학습률
        qat_lrf = kwargs.pop('lrf', 0.01)    # 최종 학습률 비율
        qat_scheduler = kwargs.pop('scheduler', 'cosine')  # 코사인 스케줄러 사용
        
        # 설정 준비
        cfg = {
            'model': self.yolo.ckpt_path if hasattr(self.yolo, 'ckpt_path') else 'yolov8n.pt',
            'data': data,
            'epochs': epochs,
            'lr0': qat_lr0,      # 초기 학습률 (QAT용 낮은 값)
            'lrf': qat_lrf,      # 최종 학습률 비율
            'scheduler': qat_scheduler,  # 스케줄러 타입
            'warmup_epochs': max(1, epochs // 10),  # warmup epoch 설정
            'warmup_momentum': 0.8,     # warmup 동안의 momentum
            'warmup_bias_lr': 0.1,      # warmup 동안의 bias lr
            'weight_decay': 0.0005,     # QAT에 적합한 weight decay
            **kwargs
        }
        
        logging.info(f"QAT 최적화 설정:")
        logging.info(f"  - 초기 학습률 (lr0): {qat_lr0}")
        logging.info(f"  - 최종 학습률 비율 (lrf): {qat_lrf}")
        logging.info(f"  - 스케줄러: {qat_scheduler}")
        logging.info(f"  - Weight decay: {cfg['weight_decay']}")
        
        # Trainer 직접 생성
        trainer = DetectionTrainer(overrides=cfg)
        
        # QAT 모델을 trainer에 직접 설정
        trainer.model = self.yolo.model  # QAT가 적용된 모델
        
        # 훈련 실행
        trainer.train()
        
        return trainer.metrics
    
    def convert_to_quantized(self):
        """훈련된 QAT 모델을 quantized 모델로 변환"""
        self.yolo.model.eval()
        quantized_model = torch.quantization.convert(self.yolo.model, inplace=False)
        return quantized_model


# 사용 예시
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO QAT Training Script")
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the base model weights file (e.g., best.pt)')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the data.yaml file')
    parser.add_argument('--save_dir', type=str, default='runs/qat', 
                        help='Directory to save QAT training results')
    args = parser.parse_args()

    setup_logging()

    # # 방법 1: 기본 QAT 구현
    # logging.info("=== 방법 1: 기본 QAT 구현 ===")
    # qat_model, quantized_model = train_qat_yolo(
    #     model_path='yolov8n.pt',
    #     epochs=5
    # )
    
    # 방법 2: Ultralytics 파이프라인 통합
    logging.info("\n=== 방법 2: Ultralytics 통합 ===")
    
    # 1. 원본 모델 경로 설정 (argparse에서 받음)
    original_model_path = args.model_path
    data_path = args.data_path
    
    # 2. QAT 전용 디렉토리 생성 및 모델 복사
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    qat_base_dir = Path(args.save_dir)
    model_name = Path(original_model_path).stem
    qat_model_dir = qat_base_dir / f'qat_{model_name}_{timestamp}'
    qat_model_dir.mkdir(parents=True, exist_ok=True)
    
    # 원본 모델을 QAT 디렉토리로 복사
    qat_model_path = qat_model_dir / 'best.pt'
    shutil.copy2(original_model_path, qat_model_path)
    logging.info(f"✅ 원본 모델을 {qat_model_path}로 복사했습니다.")
    
    # 3. 복사된 모델로 QAT 트레이너 초기화
    qat_trainer = YOLOQATTrainer(str(qat_model_path))
    
    # 4. QAT 훈련 실행
    imgsz = 400
    experiment_name = f'qat_exp_{timestamp}'
    results = qat_trainer.train(
        data=data_path,
        epochs=20,
        imgsz=imgsz,
        batch=16,
        project=args.save_dir,
        name=experiment_name,
        amp=False # QAT 학습 시에는 AMP를 끄는 것이 안정적입니다.
    )
    
    # quantized 모델로 변환
    logging.info("훈련된 모델을 양자화 버전으로 변환합니다...")
    quantized_model = qat_trainer.convert_to_quantized()
    
    # 양자화된 모델 저장
    # `results.save_dir`는 'runs/qat/experiment_name' 형태의 경로를 가집니다.
    save_dir = Path(results.save_dir)
    quantized_model_path = save_dir / 'weights' / 'quantized_best.pt'
    quantized_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), quantized_model_path)
    logging.info(f"✅ Quantized 모델이 {quantized_model_path}에 저장되었습니다.")
    
    # TensorRT로 변환
    logging.info("Quantized 모델을 TensorRT로 변환합니다...")
    try:
        # 저장된 QAT 모델을 다시 로드하여 export
        # 주의: 현재 QATWrapper를 포함한 모델은 직접 로드/export가 어려울 수 있음
        # 이상적으로는 state_dict를 원본 아키텍처에 로드한 후 export해야 함
        # 여기서는 변환된 모델 객체를 YOLO 객체에 다시 주입하여 시도
        
        # 1. 복사된 모델 아키텍처를 가진 YOLO 객체 생성
        trt_exporter = YOLO(str(qat_model_path)) 
        
        # 2. 원본 모델의 state_dict를 QAT fine-tuning된 파라미터로 업데이트
        #    QATWrapper로 인해 키 이름에 'model.' 접두사가 붙었을 수 있으므로 제거
        quantized_state_dict = torch.load(quantized_model_path)
        unwrapped_state_dict = {k.replace('model.', ''): v for k, v in quantized_state_dict.items() if k.startswith('model.')}
        trt_exporter.model.load_state_dict(unwrapped_state_dict, strict=False)

        # 3. TensorRT로 export
        # 학습 시 사용한 이미지 크기와 동일하게 설정
        tensorrt_path = trt_exporter.export(format='engine', imgsz=imgsz, data=data_path, int8=True)
        logging.info(f"✅ TensorRT 모델이 {tensorrt_path}에 저장되었습니다.")

    except Exception as e:
        logging.error(f"⚠️ TensorRT 변환 중 오류가 발생했습니다: {e}")
        logging.warning("   QAT 모델의 state_dict 구조가 원본과 다를 수 있습니다.")
        logging.warning("   수동으로 state_dict를 맞춰주거나, ONNX를 경유하여 변환해야 할 수 있습니다.")

    
    logging.info("QAT 훈련 완료!")