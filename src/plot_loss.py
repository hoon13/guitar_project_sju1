import re
import matplotlib.pyplot as plt

def generate_loss_plot(log_path, output_image_path):
    epochs = []
    train_losses = []
    val_losses = []
    
    # Pattern to parse: Epoch [01/30] Completed - Train Loss: 0.08180 | Val Loss: 0.02601
    pattern = re.compile(r"Epoch\s+\[(\d+)/\d+\]\s+Completed\s+-\s+Train\s+Loss:\s+([\d\.]+)\s+\|\s+Val\s+Loss:\s+([\d\.]+)")
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    epoch = int(match.group(1))
                    train_loss = float(match.group(2))
                    val_loss = float(match.group(3))
                    
                    epochs.append(epoch)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return
        
    if not epochs:
        print("No completed epochs found to plot.")
        return
        
    # Generate the loss curve plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#0d0e15')
    ax.set_facecolor('#0d0e15')
    
    ax.plot(epochs, train_losses, label='Train Loss', color='#ff4b2b', marker='o', lw=2)
    ax.plot(epochs, val_losses, label='Val Loss', color='#2ed573', marker='s', lw=2)
    
    ax.set_xlabel('Epoch', color='#e2e8f0', fontsize=12)
    ax.set_ylabel('Huber Loss', color='#e2e8f0', fontsize=12)
    ax.set_title('Multi-Task ResNet-18 Loss Curves', color='#e2e8f0', fontsize=14, fontweight='bold')
    
    ax.tick_params(colors='#e2e8f0')
    ax.spines['bottom'].set_color('#e2e8f0')
    ax.spines['left'].set_color('#e2e8f0')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.legend(facecolor='#0d0e15', edgecolor='white', labelcolor='white')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Force integer labels on X axis
    ax.xaxis.get_major_locator().set_params(integer=True)
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    plt.close()
    print(f"Loss plot updated at: {output_image_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        generate_loss_plot(sys.argv[1], sys.argv[2])
