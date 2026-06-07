"""
STL-10 dataset loaders for RHAN training.

STL-10 properties:
  - 96x96 RGB images (9x more pixels than CIFAR-10)
  - 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
  - 5,000 labeled training images (500/class)
  - 8,000 test images (800/class)
  - 100,000 unlabeled images (use for pretraining later)

Class mapping (same order as CIFAR-10 where applicable):
  airplane=0, bird=1, car=2, cat=3, deer=4, dog=5, horse=6, monkey=7, ship=8, truck=9
  Note: monkey replaces frog — different visual category
  Note: car replaces automobile — same category
"""

import torchvision.transforms as T
from torchvision.datasets import STL10
from torch.utils.data import DataLoader

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD  = (0.2242, 0.2215, 0.2239)

STL10_CLASSES = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'horse', 'monkey', 'ship', 'truck'
]

# Valid pixel range for RHAN attacks (same formula as CIFAR):
# min = (0 - mean) / std per channel
# max = (1 - mean) / std per channel
STL10_MIN = tuple(-(m/s) for m, s in zip(STL10_MEAN, STL10_STD))
STL10_MAX = tuple((1-m)/s for m, s in zip(STL10_MEAN, STL10_STD))


def get_stl10_loaders(batch_size=64, data_root='./data/stl10'):
    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),        # 12px = 12.5% of 96
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),  # mild augmentation
        T.ToTensor(),
        T.Normalize(STL10_MEAN, STL10_STD),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(STL10_MEAN, STL10_STD),
    ])

    train_ds = STL10(data_root, split='train',
                     transform=train_transform, download=False)
    test_ds  = STL10(data_root, split='test',
                     transform=test_transform, download=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=6,
                              pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=6,
                              pin_memory=True, persistent_workers=True)
    return train_loader, test_loader
