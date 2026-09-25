"""Datasets used in the workshop. Every image tensor is scaled to [-1, 1]."""
import os
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from . import SMOKE

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
FLOWER_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
CIFAR10_LABELS = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
]


class SyntheticImages(Dataset):
    """Tiny fake dataset (class-dependent stripes + noise) for smoke tests."""

    def __init__(self, n=256, channels=1, img_size=16, n_classes=10, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, n_classes, (n,), generator=g)
        ramp = torch.linspace(0, 3.14159 * 2, img_size)
        imgs = []
        for y in self.labels:
            wave = torch.sin(ramp * (1 + y.item() % 5))[None, :].repeat(img_size, 1)
            if y.item() >= n_classes // 2:
                wave = wave.t()
            img = wave[None].repeat(channels, 1, 1) + 0.1 * torch.randn(channels, img_size, img_size, generator=g)
            imgs.append(img.clamp(-1, 1))
        self.imgs = torch.stack(imgs)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.imgs[i], int(self.labels[i])


def _to_minus1_1(t):
    return t * 2 - 1


def _gray_transform(img_size, flip):
    from torchvision import transforms

    steps = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if flip:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.Lambda(_to_minus1_1))
    return transforms.Compose(steps)


def _loader(ds, batch_size, shuffle=True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, num_workers=0)


class _InMemory(Dataset):
    """Pre-transform a small dataset once so every epoch is fast (no PIL work per step)."""

    def __init__(self, ds, flip=False):
        loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
        xs, ys = zip(*[(x, y) for x, y in loader])
        self.x, self.y, self.flip = torch.cat(xs), torch.cat(ys), flip

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        if self.flip and torch.rand(()) < 0.5:
            x = x.flip(-1)
        return x, int(self.y[i])


def get_fashion_mnist(img_size=16, batch_size=128, root="./data"):
    """FashionMNIST (60k training images) resized to img_size. Returns (dataset, loader)."""
    if SMOKE:
        ds = SyntheticImages(256, 1, img_size, 10)
        return ds, _loader(ds, min(batch_size, 32))
    from torchvision import datasets

    raw = datasets.FashionMNIST(root, train=True, download=True, transform=_gray_transform(img_size, flip=False))
    ds = _InMemory(raw, flip=True)
    return ds, _loader(ds, batch_size)


def get_mnist(img_size=28, batch_size=128, root="./data", train=True):
    """MNIST digits in [-1, 1]. Returns (dataset, loader)."""
    if SMOKE:
        ds = SyntheticImages(256, 1, img_size, 10, seed=1 if train else 2)
        return ds, _loader(ds, min(batch_size, 32))
    from torchvision import datasets

    raw = datasets.MNIST(root, train=train, download=True, transform=_gray_transform(img_size, flip=False))
    ds = _InMemory(raw)
    return ds, _loader(ds, batch_size, shuffle=train)


# --------------------------------------------------------------------------- #
# Colour photos for the CLIP / text-to-image module
# --------------------------------------------------------------------------- #
def _fake_photo_folder(n_per_class=8, classes=("daisy", "roses", "sunflowers")):
    from PIL import Image

    root = tempfile.mkdtemp(prefix="fake_flowers_")
    g = torch.Generator().manual_seed(0)
    for k, name in enumerate(classes):
        os.makedirs(os.path.join(root, name), exist_ok=True)
        for i in range(n_per_class):
            arr = (torch.rand(48, 48, 3, generator=g) * 80).to(torch.uint8)
            arr[..., k % 3] += 150
            Image.fromarray(arr.numpy()).save(os.path.join(root, name, f"{i}.jpg"))
    return root


def get_photo_paths(name="flowers", root="./data", max_per_class=None):
    """Return (paths, labels, class_names) for a folder-per-class photo dataset.

    name="flowers": TensorFlow flower photos (3,670 JPEGs, 5 classes, ~220 MB).
    name="cifar10": CIFAR-10 exported to PNG on first use (fallback if the flower
    download is blocked on your network).
    """
    from torchvision import datasets

    if SMOKE:
        folder = _fake_photo_folder()
    elif name == "flowers":
        folder = os.path.join(root, "flower_photos")
        if not os.path.isdir(folder):
            from torchvision.datasets.utils import download_and_extract_archive

            download_and_extract_archive(FLOWER_URL, root)
    elif name == "cifar10":
        folder = os.path.join(root, "cifar10_png")
        if not os.path.isdir(folder):
            ds = datasets.CIFAR10(root, train=True, download=True)
            for i, (img, y) in enumerate(ds):
                d = os.path.join(folder, CIFAR10_LABELS[y])
                os.makedirs(d, exist_ok=True)
                img.save(os.path.join(d, f"{i}.png"))
    else:
        raise ValueError(f"unknown dataset {name!r}")

    index = datasets.ImageFolder(folder)
    paths, labels, seen = [], [], {}
    for p, y in index.samples:
        seen[y] = seen.get(y, 0) + 1
        if max_per_class is None or seen[y] <= max_per_class:
            paths.append(p)
            labels.append(y)
    return paths, labels, index.classes


def load_photo(path, img_size=32):
    """Open one photo. Returns (tensor in [-1, 1] at img_size, square PIL image)."""
    from PIL import Image
    from torchvision import transforms

    img = Image.open(path).convert("RGB")
    img = transforms.CenterCrop(min(img.size))(img)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(_to_minus1_1),
    ])
    return tf(img), img
