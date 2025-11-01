import clip

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

class clipmodel(Backbone):

    def preprocess(self, image):
        return self.preprocess(image.clone())
    
    def __init__(self, freeze=True, pretrained=True):
        super().__init__()
        self.feature_extractor, self.preprocess = clip.load("ViT-L/14", device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        # self.fc = nn.Linear(768, 2)
        # self.fc = LinearClassifier(768, 2)
        self._out_features = 768

        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        # with torch.no_grad():
        intermediate_output = self.feature_extractor.encode_image(x)
        # output = self.fc(intermediate_output)
        return intermediate_output
    
@BACKBONE_REGISTRY.register()
def clip_model(freeze=True, **kwargs):
    return clipmodel(freeze=freeze, **kwargs)