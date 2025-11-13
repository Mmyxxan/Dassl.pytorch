from .head import HEAD_REGISTRY, build_head
from .network import NETWORK_REGISTRY, build_network
from .backbone import BACKBONE_REGISTRY, Backbone, build_backbone
from .backbone.fused_backbone import FusedBackbone
from .backbone.clip_model import clipmodel
from .backbone.cnn_resnet50 import ResNet50
from .backbone.clip_vit import CLIPViT
from .backbone.clip_vit_fare import CLIPViT_FARE
