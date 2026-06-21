import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .models import get_multitask_resnet18

def evaluate_model(test_path, model_path, plot_save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError("Test data file not found.")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Trained model file not found.")
        
    X_test, y_test = torch.load(test_path)
    
    # Load model
    model = get_multitask_resnet18(num_outputs=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    y_preds = []
    y_trues = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_preds.append(outputs.cpu().numpy())
            y_trues.append(labels.numpy())
            
    y_pred = np.vstack(y_preds)
    y_true = np.vstack(y_trues)
    
    targets = ['Gain', 'Bass', 'Mid', 'Treble']
    print("\n==========================================")
    print("--- Multi-Task Tone Analyzer Final Evaluation ---")
    
    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_mse = mean_squared_error(y_true, y_pred)
    overall_r2 = r2_score(y_true, y_pred)
    
    print(f"Overall MAE: {overall_mae:.4f}")
    print(f"Overall MSE: {overall_mse:.4f}")
    print(f"Overall R2 Score: {overall_r2:.4f}")
    print("------------------------------------------")
    
    for i, target in enumerate(targets):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        print(f"{target} Task -> MAE: {mae:.4f} | MSE: {mse:.4f} | R2: {r2:.4f}")
        
    print("==========================================")
    
    # Generate scatter plot for verification
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, target in enumerate(targets):
        axes[i].scatter(y_true[:, i], y_pred[:, i], alpha=0.3, color='royalblue', edgecolors='k', s=15)
        axes[i].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[i].set_title(f"{target}: Actual vs Predicted")
        axes[i].set_xlabel("Actual")
        axes[i].set_ylabel("Predicted")
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True)
        
    plt.tight_layout()
    if plot_save_path:
        plt.savefig(plot_save_path, dpi=150)
        print(f"Saved actual vs predicted plot to {plot_save_path}")
    plt.close()
    
    return overall_mae, overall_r2
