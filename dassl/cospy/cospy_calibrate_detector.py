import torch
from torchvision import transforms
import open_clip
from diffusers import StableDiffusionPipeline
import torch
import torch.nn as nn
from torch.nn import functional as F
from io import BytesIO

import random
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter

from pathlib import Path

import cv2

def _load_ckpt(path, device="cuda" if torch.cuda.is_available() else "cpu"):
    # path can be str/Path; device is torch.device("cpu") or cuda
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Map checkpoint tensors to the requested device (CPU if no CUDA)
    map_loc = torch.device(device) if isinstance(device, str) else device
    try:
        # Safe load (PyTorch >= 2.4): avoids executing pickled code
        ckpt = torch.load(str(path), map_location=map_loc, weights_only=True)
    except TypeError:
        # Older PyTorch without weights_only
        ckpt = torch.load(str(path), map_location=map_loc)
    return ckpt

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)
def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

def jpeg_from_key(img, compress_val, key):
    jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
    method = jpeg_dict[key]
    return method(img, compress_val)

def data_augment(img, aug_config):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random.random() < aug_config["blur_prob"]:
        sig = sample_continuous(aug_config["blur_sig"])
        gaussian_blur(img, sig)

    if random.random() < aug_config["jpg_prob"]:
        method = sample_discrete(aug_config["jpg_method"])
        qual = sample_discrete(aug_config["jpg_qual"])
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class VAEReconEncoder(nn.Module):
    def __init__(self, vae, block=Bottleneck):
        super(VAEReconEncoder, self).__init__()

        # Define the ResNet model
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 is [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64 , 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        # self.layer3 = self._make_layer(block, 256, 6, stride=2)
        # self.layer4 = self._make_layer(block, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Load the VAE model
        self.vae = vae

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def reconstruct(self, x):
        with torch.no_grad():
            # `.sample()` means to sample a latent vector from the distribution
            # `.mean` means to use the mean of the distribution
            latent = self.vae.encode(x).latent_dist.mean
            decoded = self.vae.decode(latent).sample
        return decoded

    def forward(self, x):
        # Reconstruct
        x_recon = self.reconstruct(x)
        # Compute the artifacts
        x = x - x_recon

        # Scale the artifacts
        x = x / 7. * 100.

        # Forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

# Artifact Detector (Extract artifact features using VAE)
class ArtifactDetector(torch.nn.Module):
    def __init__(self, dim_artifact=512, num_classes=1):
        super(ArtifactDetector, self).__init__()
        # Load the pre-trained VAE
        model_id = "CompVis/stable-diffusion-v1-4"
        vae = StableDiffusionPipeline.from_pretrained(model_id).vae
        # Freeze the VAE visual encoder
        vae.requires_grad_(False)
        self.artifact_encoder = VAEReconEncoder(vae)

        # Classifier
        self.fc = torch.nn.Linear(dim_artifact, num_classes)

        # Normalization
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]

        # Resolution
        self.loadSize = 256
        self.cropSize = 224

        # Data augmentation
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            aug_func,
            rz_func,
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def forward(self, x, return_feat=False):
        feat = self.artifact_encoder(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        save_params = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = _load_ckpt(weights_path)
        self.load_state_dict(weights)

# Semantic Detector (Extract semantic features using CLIP)
class SemanticDetector(torch.nn.Module):
    def __init__(self, dim_clip=1152, num_classes=1):
        super(SemanticDetector, self).__init__()

        # Get the pre-trained CLIP
        model_name = "ViT-SO400M-14-SigLIP-384"
        version = "webli"
        self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=version)
        # Freeze the CLIP visual encoder
        self.clip.requires_grad_(False)

        # Classifier
        self.fc = torch.nn.Linear(dim_clip, num_classes)

        # Normalization
        self.mean = [0.5, 0.5, 0.5]
        self.std = [0.5, 0.5, 0.5]

        # Resolution
        self.loadSize = 384
        self.cropSize = 384

        # Data augmentation
        self.blur_prob = 0.5
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(30, 101))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            rz_func,
            aug_func,
            crop_func,
            flip_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def forward(self, x, return_feat=False):
        feat = self.clip.encode_image(x)
        out = self.fc(feat)
        if return_feat:
            return feat, out
        return out

    def save_weights(self, weights_path):
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = _load_ckpt(weights_path)
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]

# CO-SPY Calibrate Detector (Calibrate the integration of semantic and artifact detectors)
class CospyCalibrateDetector(torch.nn.Module):
    def __init__(self, semantic_weights_path, artifact_weights_path, num_classes=1):
        super(CospyCalibrateDetector, self).__init__()

        # Load the semantic detector
        self.sem = SemanticDetector()
        if semantic_weights_path:
            self.sem.load_weights(semantic_weights_path)

        # Load the artifact detector
        self.art = ArtifactDetector()
        if artifact_weights_path:
            self.art.load_weights(artifact_weights_path)

        # Freeze the two pre-trained models
        for param in self.sem.parameters():
            param.requires_grad = False
        for param in self.art.parameters():
            param.requires_grad = False

        # Classifier
        self.fc = torch.nn.Linear(2, num_classes)

        # Transformations inside the forward function
        # Including the normalization and resizing (only for the artifact detector)
        self.sem_transform = transforms.Compose([
            transforms.Normalize(self.sem.mean, self.sem.std)
        ])
        self.art_transform = transforms.Compose([
            transforms.Resize(self.art.cropSize, antialias=False),
            transforms.Normalize(self.art.mean, self.art.std)
        ])

        # Resolution
        self.loadSize = 384
        self.cropSize = 384

        # Data augmentation
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ['cv2', 'pil']
        self.jpg_qual = list(range(70, 96))

        # Define the augmentation configuration
        self.aug_config = {
            "blur_prob": self.blur_prob,
            "blur_sig": self.blur_sig,
            "jpg_prob": self.jpg_prob,
            "jpg_method": self.jpg_method,
            "jpg_qual": self.jpg_qual,
        }

        # Pre-processing
        crop_func = transforms.RandomCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)
        aug_func = transforms.Lambda(lambda x: data_augment(x, self.aug_config))

        self.train_transform = transforms.Compose([
            flip_func,
            aug_func,
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            rz_func,
            crop_func,
            transforms.ToTensor(),
        ])

    def forward(self, x):
        x_sem = self.sem_transform(x)
        x_art = self.art_transform(x)
        pred_sem = self.sem(x_sem)
        pred_art = self.art(x_art)
        x = torch.cat([pred_sem, pred_art], dim=1)
        x = self.fc(x)
        z = x.squeeze(1)                  # [B]
        zeros = torch.zeros_like(z)       # [B]
        logits2 = torch.stack([zeros, z], dim=1)  # [B, 2]; idx0='real', idx1='fake'
        return logits2

    def save_weights(self, weights_path):
        save_params = {"fc.weight": self.fc.weight.cpu(), "fc.bias": self.fc.bias.cpu()}
        torch.save(save_params, weights_path)

    def load_weights(self, weights_path):
        weights = _load_ckpt(weights_path)
        self.fc.weight.data = weights["fc.weight"]
        self.fc.bias.data = weights["fc.bias"]
