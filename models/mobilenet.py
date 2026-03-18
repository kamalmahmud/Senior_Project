import torch.nn as nn
import torchvision.models as models
from tsm import TemporalShift


class TSMMobileNetV2(nn.Module):
    def __init__(self, n_segment=16, num_classes=2):
        super().__init__()
        self.n_segment = n_segment

        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')

        # Insert TSM into each InvertedResidual block
        self.tsm = TemporalShift(n_segment=n_segment)

        # Keep everything except the final classifier
        self.features = backbone.features  # outputs (B*T, 1280, 4, 4)

        # Global average pool + classifier
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B*T, 1280, 4, 4) → (B*T, 1280, 1, 1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        # x: (B, 3, T, 112, 112)
        B, C, T, H, W = x.shape

        # Step 1: reshape for 2D backbone
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)  # (B*T, C, H, W)

        # Step 2: apply TSM then backbone
        x = self.tsm(x)  # temporal shift
        x = self.features(x)  # (B*T, 1280, 4, 4)

        # Step 3: spatial pooling
        x = self.pool(x)  # (B*T, 1280, 1, 1)
        x = x.flatten(1)  # (B*T, 1280)

        # Step 4: fold back and average across T
        x = x.reshape(B, T, -1)  # (B, T, 1280)
        x = x.mean(dim=1)  # (B, 1280)

        # Step 5: classify
        x = self.classifier(x)  # (B, 2)
        return x
