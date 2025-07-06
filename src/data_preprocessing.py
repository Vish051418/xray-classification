import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class XRayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def load_and_preprocess_data(data_dir, img_size=(224, 224), test_size=0.2):
    """Load and preprocess X-ray images"""
    classes = ["COVID19", "NORMAL", "PNEUMONIA"]
    images = []
    labels = []
    
    print("Starting data preprocessing...")
    
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        print(f"Processing class: {class_name}")
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            
            images.append(img)
            labels.append(classes.index(class_name))
    
    images = np.array(images)
    labels = np.array(labels)
    
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, 
        test_size=test_size, 
        stratify=labels,
        random_state=42
    )
    
    os.makedirs("data/processed", exist_ok=True)
    np.savez("data/processed/train.npz", X=X_train, y=y_train)
    np.savez("data/processed/val.npz", X=X_val, y=y_val)
    
    print(f"Data saved to: data/processed/")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val

def get_dataloaders(batch_size=32, num_workers=0):  # Added num_workers parameter
    """Create PyTorch dataloaders"""
    train_data = np.load("data/processed/train.npz")
    val_data = np.load("data/processed/val.npz")
    
    train_dataset = XRayDataset(train_data['X'], train_data['y'])
    val_dataset = XRayDataset(val_data['X'], val_data['y'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers  # Now accepts this parameter
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers  # Now accepts this parameter
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    load_and_preprocess_data("data/train")