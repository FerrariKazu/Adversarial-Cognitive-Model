"""
STL-10 dataset loaders for RHAN training.

STL-10 properties:
  - 96x96 RGB images (9x more pixels than CIFAR-10)
  - 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck
  - 5,000 labeled training images (500/class)
  - 8,000 test images (800/class)
  - 100,000 unlabeled images for semi-supervised pretraining
"""

import torchvision.transforms as T
from torchvision.datasets import STL10 as _STL10
from torch.utils.data import DataLoader, ConcatDataset

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD  = (0.2242, 0.2215, 0.2239)

STL10_CLASSES = [
    'airplane', 'bird', 'car', 'cat', 'deer',
    'dog', 'horse', 'monkey', 'ship', 'truck'
]

# Valid pixel range for RHAN attacks
STL10_MIN = tuple(-(m/s) for m, s in zip(STL10_MEAN, STL10_STD))
STL10_MAX = tuple((1-m)/s for m, s in zip(STL10_MEAN, STL10_STD))


class STL10(_STL10):
    """STL10 that skips unlabeled file integrity check."""

    def _check_integrity(self) -> bool:
        import os, hashlib
        for filename, md5 in self.train_list[:2] + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not os.path.exists(fpath):
                return False
            actual = hashlib.md5(open(fpath, 'rb').read()).hexdigest()
            if actual != md5:
                return False
        return True


def get_stl10_loaders(batch_size=64, data_root='./data/stl10'):
    """Labeled train/test loaders."""
    train_transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
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
                              shuffle=True, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=4,
                              pin_memory=True, persistent_workers=True)
    return train_loader, test_loader


def get_stl10_unlabeled_loader(batch_size=128, data_root='./data/stl10'):
    """100K unlabeled images for self-supervised pretraining."""
    transform = T.Compose([
        T.RandomCrop(96, padding=12),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.15),
        T.ToTensor(),
        T.Normalize(STL10_MEAN, STL10_STD),
    ])
    unlabeled_ds = STL10(data_root, split='unlabeled',
                         transform=transform, download=False)
    return DataLoader(unlabeled_ds, batch_size=batch_size,
                      shuffle=True, num_workers=6,
                      pin_memory=True, persistent_workers=True)
