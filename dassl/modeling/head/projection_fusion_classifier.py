import torch
import torch.nn as nn

from .build import HEAD_REGISTRY

class PFC(nn.Module):
    def __init__(self, num_classes=2, resnet_out_dim=2048, vit_out_dim=1024, project_dim=512):
        super().__init__()
        self.resnet_proj = nn.Linear(resnet_out_dim, project_dim)
        self.vit_proj = nn.Linear(vit_out_dim, project_dim)

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2*project_dim, num_classes)
        )

    def forward(self, resnet_out, vit_out):
        fused = torch.cat([self.resnet_proj(resnet_out), self.vit_proj(vit_out)], dim=1) # (B, 1024)
        return self.classifier(fused)

@HEAD_REGISTRY.register()
def pfc(**kwargs):
    return PFC(**kwargs)