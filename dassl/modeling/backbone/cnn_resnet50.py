import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

# shared_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
#     transforms.ToTensor()
# ])

class ResNet50(Backbone):

    resnet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    @staticmethod
    def preprocess(image_tensor):
        return ResNet50.resnet_normalize(image_tensor.clone())  # Clone to avoid in-place ops

    def __init__(self, freeze=True):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self._out_features = 2048

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

@BACKBONE_REGISTRY.register()
def cnn_resnet50(freeze=True, **kwargs):
    model = ResNet50(freeze=freeze, **kwargs)
    return model