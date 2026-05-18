import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from PIL import Image

# -----------------------------------------------------------------------------
# CIFAR-10 Constants
# -----------------------------------------------------------------------------
# 1. WHAT: Defining the standard CIFAR-10 class names.
# 2. WHY: Useful for visualization and per-class evaluation metrics later.
# 3. OBSERVE: Prints/plots will show "cat" instead of class index "3".
# -----------------------------------------------------------------------------
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# -----------------------------------------------------------------------------
# HuggingFace Dataset Wrapper
# -----------------------------------------------------------------------------
# 1. WHAT: Wraps the HuggingFace `datasets` CIFAR-10 into a PyTorch Dataset.
# 2. WHY: The official University of Toronto CIFAR-10 server often goes offline
#         (HTTP 503 errors). HuggingFace's infrastructure is significantly more
#         robust and bypasses this downtime entirely.
# 3. OBSERVE: Downloads the dataset seamlessly via HF infrastructure.
# -----------------------------------------------------------------------------
class HFCIFAR10(Dataset):
    def __init__(self, split='train', transform=None):
        self.dataset = load_dataset('cifar10', split=split)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = item['img'].convert('RGB')
        label = item['label']
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloaders(batch_size=128, num_workers=4, data_dir='data', model_name='resnet'):
    """
    Returns DataLoaders with dynamically injected transforms based on the model architecture.
    Default fallback is the standard ResNet pipeline.
    """
    model_name = model_name.lower()
    
    # -------------------------------------------------------------------------
    # Dynamic Data Augmentation & Normalization
    # -------------------------------------------------------------------------
    if model_name in ['efficientnet', 'vit']:
        # EfficientNet & ViT: Require 224x224 and standard CIFAR-10 normalization
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    elif model_name == 'shaperesnet':
        # ShapeResNet: Uses custom transforms (trained on 64x64) as per train_shaperesnet_v2.py
        transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.3),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    elif model_name == 'bagnet':
        # BagNet: Requires at least 33x33, team standard is 64x64
        transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
    else:
        # Standard CIFAR-10 baseline: ResNet-18 & Shape-ResNet
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    # -------------------------------------------------------------------------
    # Dataset Loading (HuggingFace)
    # -------------------------------------------------------------------------
    trainset = HFCIFAR10(split='train', transform=transform_train)
    testset = HFCIFAR10(split='test', transform=transform_test)

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader