"""PIL-space image augmentation applied before pretrained-model normalization."""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image

PILTransform = Callable[[Image.Image], Image.Image]


def apply_pil_transform(
    image: Image.Image,
    transform: PILTransform | None,
) -> Image.Image:
    """Apply an augmentation and enforce that normalization has not happened yet."""

    if not isinstance(image, Image.Image):
        raise TypeError("PIL augmentation expects a PIL.Image")
    transformed = image if transform is None else transform(image)
    if not isinstance(transformed, Image.Image):
        raise TypeError(
            "pil_transform must return a PIL.Image; tensor conversion and normalization "
            "belong to the image processor after augmentation"
        )
    return transformed


def build_train_augmentation(
    *,
    horizontal_flip_probability: float = 0.4,
    rotation_degrees: float = 10.0,
    color_jitter_probability: float = 0.3,
    brightness: float = 0.3,
    hue: float = 0.1,
) -> PILTransform:
    """Build the notebook's augmentations in a safe PIL-only order.

    The returned callable deliberately excludes resize/rescale/normalization;
    those operations remain owned by the pretrained encoder's image processor.
    """

    for name, value in {
        "horizontal_flip_probability": horizontal_flip_probability,
        "color_jitter_probability": color_jitter_probability,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if rotation_degrees < 0:
        raise ValueError("rotation_degrees cannot be negative")
    if brightness < 0:
        raise ValueError("brightness cannot be negative")
    if not 0.0 <= hue <= 0.5:
        raise ValueError("hue must be in [0, 0.5]")

    # Import lazily so record/split tooling does not require torchvision to be
    # importable when no augmentation is requested.
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=horizontal_flip_probability),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=brightness, hue=hue)],
                p=color_jitter_probability,
            ),
        ]
    )
