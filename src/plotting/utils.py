import numpy as np


def rerange_image(img: np.ndarray) -> np.ndarray:
    """Rerange image to 0-255"""
    img = img - np.min(img)
    img = img / np.max(img)
    img = img * 255
    return img.astype(np.uint8)
