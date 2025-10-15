from transformers import CLIPVisionModel, CLIPImageProcessor
from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from torchvision import transforms

class CLIPViT(Backbone):

    clip_normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    @staticmethod
    def preprocess(image_tensor):
        return CLIPViT.clip_normalize(image_tensor.clone())

    def __init__(self, model_name="openai/clip-vit-large-patch14", freeze=True):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        self._out_features = self.model.config.hidden_size # usually 1024 for large

    def forward(self, x, return_all_tokens=False):
        outputs = self.model(pixel_values=x)
        if return_all_tokens:
            return outputs.last_hidden_state
        return outputs.last_hidden_state[:, 0] # CLS token

@BACKBONE_REGISTRY.register()
def clip_vit(freeze=True, **kwargs):
    return CLIPViT(freeze=freeze, **kwargs)
