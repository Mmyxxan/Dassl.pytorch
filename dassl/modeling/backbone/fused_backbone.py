from .cnn_resnet50 import ResNet50
from .clip_vit import CLIPViT
from .clip_vit_fare import CLIPViT_FARE
from .build import BACKBONE_REGISTRY
from .backbone import Backbone

import torch.nn as nn
import torch

class FusedBackbone(Backbone):

    @staticmethod
    def preprocess(shared_tensor):
        # print("+ normalization per backbone:")
        # for backbone_cls in FusedBackbone.backbone_list:
        #     print(f"  - {backbone_cls.__name__}")
        return [backbone_cls.preprocess(shared_tensor) for backbone_cls in FusedBackbone.backbone_list]

    def __init__(self, backbone_list, freeze=True, project_dim=512, pretrained=True, **kwargs):
        super().__init__()
        self.backbone_list = backbone_list

        self.backbones = nn.ModuleList()
        self.projections = nn.ModuleList()

        for backbone_cls in self.backbone_list:
            model = backbone_cls(freeze=freeze, pretrained=pretrained)
            projection = nn.Linear(model._out_features, project_dim)
            self.backbones.append(model)
            self.projections.append(projection)

        self._out_features = project_dim * len(self.backbone_list)

    def forward(self, inputs):
        assert len(inputs) == len(self.backbones)
        outputs = []
        for i, input in enumerate(inputs):
            outputs.append(self.projections[i](self.backbones[i](input)))
        return torch.cat(outputs, dim=1)
    
@BACKBONE_REGISTRY.register()
def fused_cnn_resnet50_clip_vit(freeze=True, **kwargs):
    return FusedBackbone(backbone_list=[ResNet50, CLIPViT], freeze=freeze, **kwargs)

@BACKBONE_REGISTRY.register()
def fused_cnn_resnet50_robust_clip_vit(freeze=True, **kwargs):
    return FusedBackbone(backbone_list=[ResNet50, CLIPViT_FARE], freeze=freeze, **kwargs)