import torch 
from torch.utils.data import DataLoader, random_split
import timm 
from pathlib import Path 

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
    dataset = CustomImageFolder(root=data_src, 
                                is_valid_file=CustomImageFolder.is_valid_image_file
                                )
    
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
                 ) -> float:
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
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

def _valid_epoch(model: torch.nn.Module,
                 valid_loader: DataLoader,
                 criterion: torch.nn.Module,
                 device: str
                 ) -> float:
    """
    하나의 에포크를 검증하는 함수 
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
    return epoch_loss

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
    
    best_valid_loss = float('inf')
    
    weights_dir: Path = Path("models")
    weights_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss = _train_epoch(model=model, 
                                  train_loader=train_loader, 
                                  optimizer=optimizer, 
                                  criterion=criterion, 
                                  device=device
                                  )
        
        valid_loss = _valid_epoch(model=model, 
                                  valid_loader=valid_loader,
                                  criterion=criterion,
                                  device=device
                                  )
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), weights_dir / "best.pt")
            print(f"Best model saved with valid loss: {best_valid_loss:.4f}")

    torch.save(model.state_dict(), weights_dir / "latest.pt")
    print("Latest model saved.")

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