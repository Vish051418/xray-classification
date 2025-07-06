import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import get_dataloaders
import os

class XRayCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(XRayCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            
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
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        if scheduler:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"Learning rate reduced from {old_lr:.2e} to {new_lr:.2e}")
                mlflow.log_metric("learning_rate", new_lr, step=epoch)
        
        mlflow.log_metrics({
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'train_acc': epoch_acc,
            'val_acc': val_acc
        }, step=epoch)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "models/best_model.pth")
            mlflow.log_artifact("models/best_model.pth")
            print(f"New best model saved with val_acc: {best_acc:.4f}")
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, train_accs, val_accs

def evaluate_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.float()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    loss = running_loss / len(loader.dataset)
    acc = correct / total
    
    report = classification_report(
        all_labels, all_preds,
        target_names=["COVID19", "NORMAL", "PNEUMONIA"],
        output_dict=True
    )
    mlflow.log_dict(report, "classification_report.json")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', 
               xticklabels=["COVID19", "NORMAL", "PNEUMONIA"],
               yticklabels=["COVID19", "NORMAL", "PNEUMONIA"])
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()
    
    return loss, acc

def main():
    # Set up MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Xray_Classification")
    
    with mlflow.start_run():
        # Hyperparameters
        params = {
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 50,
            "weight_decay": 1e-5,
            "min_lr": 1e-6,
            "patience": 3,
            "factor": 0.1
        }
        mlflow.log_params(params)
        
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mlflow.log_param("device", str(device))
        
        # Get data loaders
        train_loader, val_loader = get_dataloaders(params["batch_size"], num_workers=0)
        
        # Initialize model
        model = XRayCNN(num_classes=3).to(device)
        
        # Create proper input example (convert tensor to numpy array)
        input_example = torch.rand(1, 3, 224, 224).to(device).cpu().numpy()
        
        # Infer model signature
        with torch.no_grad():
            output_example = model(torch.from_numpy(input_example).to(device)).cpu().numpy()
        signature = mlflow.models.infer_signature(input_example, output_example)
        
        # Log model with all required parameters
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            conda_env={
                'channels': ['defaults'],
                'dependencies': [
                    'python=3.8',
                    'pytorch=1.12.1',
                    'torchvision=0.13.1',
                    'mlflow=1.26.1',
                    {'pip': ['numpy==1.23.5']}
                ],
                'name': 'xray_env'
            },
            code_paths=[__file__],
            registered_model_name="Xray_Classifier",
            signature=signature,
            input_example=input_example
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), 
                             lr=params["learning_rate"],
                             weight_decay=params["weight_decay"])
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=params["factor"],
            patience=params["patience"],
            min_lr=params["min_lr"]
        )
        
        # Train model
        train_losses, val_losses, train_accs, val_accs = train_model(
            model, train_loader, val_loader, criterion, 
            optimizer, scheduler, params["num_epochs"], device
        )
        
        # Plot training curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.savefig("training_curves.png")
        mlflow.log_artifact("training_curves.png")
        plt.close()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Set multiprocessing start method for Windows compatibility
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    main()