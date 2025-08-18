import torch 
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
import timm 
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import io

from config import classes
from utils import CustomImageFolder


def load_model(model_name: str,
               classes: list[str]
               ) -> torch.nn.Module:
    """
    timm 라이브러리를 사용하여 모델을 로드합니다.
    """
    model = timm.create_model(model_name, 
                              pretrained=True, 
                              num_classes=len(classes)
                              )
    return model 

def load_data(data_src: str | Path,
              batch_size: int = 16,
              train_split_ratio: float = 0.8
              ) -> tuple[DataLoader, DataLoader]:
    """
    torchvision.datasets.ImageFolder를 사용하여 데이터를 로드합니다.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = CustomImageFolder(root=data_src, 
                              transform=transform,
                              is_valid_file=CustomImageFolder.is_valid_image_file)
    
    # 데이터셋 분할
    train_size = int(train_split_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    
    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4
                            )
    
    valid_loader = DataLoader(dataset=valid_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4
                            )
    return train_loader, valid_loader

def _train_epoch(model: torch.nn.Module,
                 train_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: str
                 ) -> tuple[float, float]:
    """
    하나의 에포크를 학습하는 함수 

    Args:
        model: 학습할 모델 
        train_loader: 훈련 데이터셋 
        optimizer: 최적화 함수 
        criterion: 손실 함수 
        device: 사용할 디바이스 
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def _valid_epoch(model: torch.nn.Module,
                 valid_loader: DataLoader,
                 criterion: torch.nn.Module,
                 device: str
                 ) -> tuple[float, float, list, list]:
    """
    하나의 에포크를 검증하는 함수 
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_confusion_matrix(preds, labels, class_names):
    """Confusion matrix를 생성하고 이미지로 반환합니다."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    image = transforms.ToTensor()(plt.imread(buf, format='png'))
    return image


def train_model(model: torch.nn.Module,
                train_loader: DataLoader,
                valid_loader: DataLoader,
                epochs: int = 100,
                learning_rate: float = 0.001,
                device: int = 0
                ) -> None:
    """
    모델을 학습하는 함수 

    Args:
        model: 학습할 모델 
        train_loader: 훈련 데이터셋 
        valid_loader: 검증 데이터셋 
        epochs: 학습 에포크 수 
        learning_rate: 학습률 
        device: 사용할 디바이스 
    """
    device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    writer = SummaryWriter('runs/model_experiment')
    
    best_valid_loss = float('inf')
    
    weights_dir: Path = Path("models")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # 고정된 검증 이미지 배치 가져오기
    fixed_val_images, fixed_val_labels = next(iter(valid_loader))

    for epoch in range(epochs):
        train_loss, train_acc = _train_epoch(model=model, 
                                             train_loader=train_loader, 
                                             optimizer=optimizer, 
                                             criterion=criterion, 
                                             device=device
                                             )
        
        valid_loss, valid_acc, valid_preds, valid_labels = _valid_epoch(model=model, 
                                                                        valid_loader=valid_loader,
                                                                        criterion=criterion,
                                                                        device=device
                                                                        )
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)

        # Confusion Matrix 로깅
        cm_image = plot_confusion_matrix(valid_preds, valid_labels, classes)
        writer.add_image('Confusion Matrix', cm_image, epoch)
        
        # 예측 결과 이미지 로깅
        model.eval()
        with torch.no_grad():
            outputs = model(fixed_val_images.to(device))
            _, predicted = torch.max(outputs, 1)
        
        img_grid = utils.make_grid(fixed_val_images)
        # 이미지에 정답/예측 텍스트 추가 (이 부분은 torchvision만으로는 복잡하여 캡션으로 대체)
        writer.add_image('Validation Predictions', img_grid, epoch)
        
        caption = "Pred: " + ", ".join(f"{classes[p]}" for p in predicted) + "\n\nTrue: " + ", ".join(f"{classes[l]}" for l in fixed_val_labels)
        writer.add_text('Prediction vs. True', caption, epoch)


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), weights_dir / "best.pt")
            print(f"Best model saved with valid loss: {best_valid_loss:.4f}")

    torch.save(model.state_dict(), weights_dir / "latest.pt")
    print("Latest model saved.")
    writer.close()

def main():
    model_name = "resnet101"
    data_src: Path = Path("data/")
    train_loader, valid_loader = load_data(data_src=data_src)
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(valid_loader.dataset)}")
    
    model = load_model(model_name=model_name, 
                       classes=classes
                       )
    
    train_model(model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                device=3
                )

if __name__ == "__main__":
    main()