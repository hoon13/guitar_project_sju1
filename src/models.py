import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadResNet18(nn.Module):
    def __init__(self, num_outputs=4):
        super().__init__()
        # Load base ResNet-18
        base_model = models.resnet18(weights=None)
        
        # Modify conv1 to accept 1 channel (Mel Spectrogram) instead of 3
        base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract the shared backbone layers
        self.shared_backbone = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool
        )
        
        # Separate heads for Gain, Bass, Mid, Treble to decouple predictions (prevent crosstalk)
        self.gain_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.bass_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.mid_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        self.treble_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        features = self.shared_backbone(x)
        features = torch.flatten(features, 1)
        
        gain = self.gain_head(features)
        bass = self.bass_head(features)
        mid = self.mid_head(features)
        treble = self.treble_head(features)
        
        # Concatenate task outputs into shape [batch_size, 4]
        return torch.cat([gain, bass, mid, treble], dim=1)

def get_multitask_resnet18(num_outputs=4):
    """
    Returns a customized ResNet-18 model with separate heads for multi-task regression.
    """
    return MultiHeadResNet18(num_outputs=num_outputs)
