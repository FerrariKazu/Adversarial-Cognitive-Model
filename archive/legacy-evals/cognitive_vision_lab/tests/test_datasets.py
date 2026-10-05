"""Unit tests for cognitive_vision_lab.backend.datasets."""
from cognitive_vision_lab.backend.datasets import (
    STL10_CLASSES,
    get_dataset,
    list_datasets,
    sample_images,
)


class TestDatasets:
    def test_known_datasets(self):
        names = list_datasets()
        for required in ["STL-10", "CIFAR-10", "CIFAR-100", "ImageNet", "ImageNet-C"]:
            assert required in names

    def test_stl10_metadata(self):
        info = get_dataset("STL-10")
        assert info.n_classes == 10
        assert info.train_size == 5000
        assert info.test_size == 8000
        assert info.classes == STL10_CLASSES

    def test_unknown_falls_back_to_stl10(self):
        info = get_dataset("Nope")
        assert info.name == "STL-10"

    def test_imagenet_c_corruptions(self):
        info = get_dataset("ImageNet-C")
        assert len(info.corruptions) >= 10
        assert "gaussian_noise" in info.corruptions

    def test_sample_images_deterministic(self):
        imgs = sample_images("STL-10", n=6)
        assert len(imgs) == 6
        for img, label in imgs:
            assert label in STL10_CLASSES
            assert img.size == (96, 96)
