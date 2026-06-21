import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from .models import get_multitask_resnet18

def train_model(train_path, test_path, save_path, epochs=30, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError("Train or test data file (.pt) not found. Run dataset generation first.")
        
    # Load dataset
    X_train, y_train = torch.load(train_path)
    X_test, y_test = torch.load(test_path)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # Instantiate customized ResNet-18
    model = get_multitask_resnet18(num_outputs=4).to(device)
    
    # Huber Loss for robust regression
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nStarting Multi-Task ResNet-18 training for {epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # [Train mode]
        model.train()
        train_loss = 0.0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}] Train", leave=False)
        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_iterator.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader.dataset)
        
        # [Eval mode]
        model.eval()
        val_loss = 0.0
        
        val_iterator = tqdm(test_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}] Valid", leave=False)
        with torch.no_grad():
            for inputs, labels in val_iterator:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(test_loader.dataset)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] Completed - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f" -> Checkpoint saved! Best Val Loss: {best_val_loss:.5f}")
            
    print("\nTraining completed successfully!")

def finetune_model(pretrained_model_path, real_train_path, real_val_path, save_path, epochs=15, batch_size=16, lr=0.0001):
    """
    Performs fine-tuning on real-world recordings.
    Loads pre-trained weights, trains with a very low learning rate (lr=0.0001)
    to adjust parameter boundaries to the user's specific setup, and saves the best model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Fine-tuning on device: {device}")
    
    if not (os.path.exists(real_train_path) and os.path.exists(real_val_path)):
        raise FileNotFoundError("Real train or validation data file (.pt) not found. Run real dataset generation first.")
        
    # Load dataset
    X_train, y_train = torch.load(real_train_path)
    X_val, y_val = torch.load(real_val_path)
    
    print(f"Loaded Real Train dataset size: {X_train.shape[0]} samples")
    print(f"Loaded Real Val dataset size: {X_val.shape[0]} samples")
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    
    # Load model and inject pre-trained weights
    model = get_multitask_resnet18(num_outputs=4)
    if os.path.exists(pretrained_model_path):
        print(f"Loading pre-trained synthetic weights from {pretrained_model_path}...")
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    else:
        print("Warning: Pre-trained synthetic model path not found. Fine-tuning from scratch...")
        
    model = model.to(device)
    
    # Huber Loss for robust regression
    criterion = nn.HuberLoss()
    # Low learning rate for fine-tuning to prevent destroying pre-trained weights
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    print(f"\nStarting Fine-tuning on Real Data for {epochs} epochs (LR={lr})...")
    
    for epoch in range(epochs):
        # [Train mode]
        model.train()
        train_loss = 0.0
        
        train_iterator = tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}] FT Train", leave=False)
        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_iterator.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader.dataset)
        
        # [Eval mode]
        model.eval()
        val_loss = 0.0
        
        val_iterator = tqdm(val_loader, desc=f"Epoch [{epoch+1:02d}/{epochs}] FT Valid", leave=False)
        with torch.no_grad():
            for inputs, labels in val_iterator:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] FT Completed - Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Create a backup of the original before overwriting (if not backed up yet)
            backup_path = pretrained_model_path.replace(".pth", "_synthetic_backup.pth")
            if not os.path.exists(backup_path) and os.path.exists(pretrained_model_path):
                import shutil
                shutil.copy2(pretrained_model_path, backup_path)
                print(f" -> Backed up original synthetic model to: {backup_path}")
                
            torch.save(model.state_dict(), save_path)
            print(f" -> Checkpoint saved! Best Real Val Loss: {best_val_loss:.5f}")
            
    print("\nFine-tuning completed successfully!")

