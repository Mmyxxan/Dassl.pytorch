import torch.nn as nn

from .build import HEAD_REGISTRY

class BinaryClassifier(nn.Module):
    def __init__(self, in_features, num_classes=2, drop_out=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(drop_out),
            nn.Linear(in_features, num_classes),
        )
        # how to decrease dependency on one feature extractor by adjusting drop-out?

    def forward(self, x):
        return self.classifier(x)
    
@HEAD_REGISTRY.register()
def binary_classifier(**kwargs):
    return BinaryClassifier(**kwargs)
