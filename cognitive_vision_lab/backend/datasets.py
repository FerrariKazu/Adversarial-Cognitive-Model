"""Dataset metadata + sample browsing for the Dataset Explorer page."""
from __future__ import annotations

from dataclasses import dataclass, field

from cognitive_vision_lab.utils.io import procedural_sample


@dataclass
class DatasetInfo:
    name: str
    n_classes: int
    resolution: str
    train_size: int
    test_size: int
    classes: list[str]
    corruptions: list[str] = field(default_factory=list)
    notes: str = ""


STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR100_SUPERCLASSES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes", "large omnivores and herbivores",
    "medium-sized mammals", "non-insect invertebrates", "people", "reptiles", "small mammals",
    "trees", "vehicles 1", "vehicles 2",
]

DATASETS: dict[str, DatasetInfo] = {
    "STL-10": DatasetInfo(
        name="STL-10", n_classes=10, resolution="96×96", train_size=5000,
        test_size=8000, classes=STL10_CLASSES,
        notes="The project's primary benchmark. 500 labeled + 100K unlabeled train images; "
              "only 10 classes, each heavily under-represented in labels.",
    ),
    "CIFAR-10": DatasetInfo(
        name="CIFAR-10", n_classes=10, resolution="32×32", train_size=50000,
        test_size=10000, classes=CIFAR10_CLASSES,
        notes="Standard research benchmark; automobile/truck collapse under AutoAttack "
              "is a dataset-intrinsic geometry problem (Finding 9/12).",
    ),
    "CIFAR-100": DatasetInfo(
        name="CIFAR-100", n_classes=100, resolution="32×32", train_size=50000,
        test_size=10000, classes=[f"class {i}" for i in range(100)],
        notes="100 fine-grained classes grouped into 20 superclasses.",
    ),
    "Tiny ImageNet": DatasetInfo(
        name="Tiny ImageNet", n_classes=200, resolution="64×64", train_size=100000,
        test_size=10000, classes=[f"class {i}" for i in range(200)],
        notes="Downsampled ImageNet; 500 train images per class.",
    ),
    "ImageNet": DatasetInfo(
        name="ImageNet", n_classes=1000, resolution="~469×387", train_size=1281167,
        test_size=50000, classes=[f"class {i}" for i in range(1000)],
        notes="De-facto standard for large-scale vision.",
    ),
    "ImageNet-C": DatasetInfo(
        name="ImageNet-C", n_classes=1000, resolution="224×224", train_size=0,
        test_size=75000,
        classes=[f"class {i}" for i in range(1000)],
        corruptions=[
            "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur",
            "motion_blur", "zoom_blur", "snow", "frost", "fog", "brightness", "contrast",
            "elastic_transform", "pixelate", "jpeg_compression",
        ],
        notes="Corruption robustness (common corruptions, 5 severities).",
    ),
    "ImageNet-A": DatasetInfo(
        name="ImageNet-A", n_classes=200, resolution="~224×224", train_size=0,
        test_size=7500, classes=[f"class {i}" for i in range(200)],
        notes="Natural adversarial examples that fool standard models.",
    ),
    "ImageNet-R": DatasetInfo(
        name="ImageNet-R", n_classes=200, resolution="~224×224", train_size=0,
        test_size=30000, classes=[f"class {i}" for i in range(200)],
        notes="Renditions: sketches, paintings, cartoons, art forms.",
    ),
}


def list_datasets() -> list[str]:
    return list(DATASETS.keys())


def get_dataset(name: str) -> DatasetInfo:
    return DATASETS.get(name, DATASETS["STL-10"])


def sample_images(dataset: str, n: int = 8) -> list:
    """Deterministic sample images (procedural fallback; real cache optional)."""
    info = get_dataset(dataset)
    imgs = []
    for i in range(n):
        label = info.classes[i % len(info.classes)] if info.classes else f"class {i}"
        imgs.append((procedural_sample(label), label))
    return imgs
