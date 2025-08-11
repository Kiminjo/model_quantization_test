import torch 
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, utils
import timm 
from pathlib import Path 
from torch.utils.tensorboard import SummaryWriter

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
                 ) -> tuple[float, float]:
    """
    하나의 에포크를 검증하는 함수 
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(valid_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

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

    for epoch in range(epochs):
        train_loss, train_acc = _train_epoch(model=model, 
                                             train_loader=train_loader, 
                                             optimizer=optimizer, 
                                             criterion=criterion, 
                                             device=device
                                             )
        
        valid_loss, valid_acc = _valid_epoch(model=model, 
                                             valid_loader=valid_loader,
                                             criterion=criterion,
                                             device=device
                                             )
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), weights_dir / "best.pt")
            print(f"Best model saved with valid loss: {best_valid_loss:.4f}")

    dataiter = iter(valid_loader)
    images, labels = next(dataiter)
    
    images_for_grid = images[:4]
    img_grid = utils.make_grid(images_for_grid)
    writer.add_image('Validation images', img_grid)
    
    model.eval()
    images_for_grid = images_for_grid.to(device)
    outputs = model(images_for_grid)
    _, predicted = torch.max(outputs, 1)
    
    class_names = valid_loader.dataset.dataset.classes
    
    writer.add_text('Predictions', 
                    'GroundTruth: ' + ' '.join(f'{class_names[labels[j]]}' for j in range(4)) + 
                    '\n\nPrediction: ' + ' '.join(f'{class_names[predicted[j]]}' for j in range(4)), 0)


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
                valid_loader=valid_loader
                )

if __name__ == "__main__":
    main()