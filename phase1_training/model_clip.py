# pip install git+https://github.com/openai/CLIP.git
"""
What CLIP is and Zero-Shot Classification:
CLIP (Contrastive Language-Image Pretraining) is a powerful vision-language model trained 
on hundreds of millions of image-text pairs. Unlike traditional supervised models trained 
to predict a fixed set of classes (e.g., via one-hot labels), CLIP learns a joint multi-modal 
embedding space. "Zero-shot" classification allows the model to classify images into categories 
it was never explicitly fine-tuned on, simply by measuring the cosine similarity between the 
encoded image embedding and the text embeddings of the target categories.

Why CLIP is scientifically important for our study:
Standard CNNs and ViTs are explicitly trained on ImageNet/CIFAR-10 and typically develop 
strong texture biases to minimize cross-entropy loss. CLIP, conversely, is trained contrastively 
with natural language, forcing it to learn more robust, semantically grounded representations. 
Evaluating CLIP's zero-shot adversarial robustness helps us understand whether language-guided 
representation learning provides a structural defense against the high-frequency pixel 
perturbations that easily shatter traditional unimodal models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import clip

class CIFARClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load CLIP ViT-B/32
        self.model, _ = clip.load('ViT-B/32', device=self.device)
        self.model.eval()
        
        # Tokenize the exact 10 prompts
        prompts = [
            'a photo of an airplane',
            'a photo of an automobile',
            'a photo of a bird',
            'a photo of a cat',
            'a photo of a deer',
            'a photo of a dog',
            'a photo of a frog',
            'a photo of a horse',
            'a photo of a ship',
            'a photo of a truck'
        ]
        text_tokens = clip.tokenize(prompts).to(self.device)
        
        # Pre-compute and store text features
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        # CLIP's specific normalization parameters
        self.normalize = transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        )
        
        # CIFAR-10 un-normalization parameters
        self.register_buffer('cifar_mean', torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer('cifar_std', torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1))

    def predict(self, images):
        # 1. Revert CIFAR-10 normalization to get back to [0, 1] range
        images = images * self.cifar_std + self.cifar_mean
        
        # 2. Resize to 224x224 and normalize using CLIP's specific mean/std
        images = F.interpolate(images, size=(224, 224), mode='bicubic', align_corners=False)
        images = self.normalize(images)
        
        # DO NOT use torch.no_grad() here, otherwise adversarial attacks (PGD) will fail!
        image_features = self.model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
            
        return logits
        
    def __call__(self, x):
        """Wraps predict() so it works like a standard PyTorch model."""
        return self.predict(x)
