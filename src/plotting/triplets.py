from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from plotting.utils import rerange_image


def plot_triplets(
    result_path: str,
    anchor_imgs: np.ndarray,
    positive_imgs: np.ndarray,
    epoch: int,
    triplet_indices: List[Tuple[int, int, int]],
):
    """
    Plot the triplets in a 3D space
    Args:
        result_path (str): Path to save the plot
        anchor_imgs (np.ndarray): The anchor images
        positive_img (np.ndarray): The positive images
        epoch (int): The epoch number
        triplet_indices (List[Tuple[int, int, int]]): The triplet indices
    """
    n_triplets = min(20, len(triplet_indices))
    fig, axs = plt.subplots(n_triplets, 3, figsize=(15, n_triplets * 5))
    for triplet_idx, triplet in enumerate(triplet_indices):
        if triplet_idx == n_triplets:
            break
        anchor_idx, negative_idx, positive_idx = triplet

        if anchor_imgs.ndim == 4:
            anchor_img = anchor_imgs[anchor_idx, 0, :]
            positive_img = positive_imgs[positive_idx, 0, :]
            negative_img = anchor_imgs[negative_idx, 0, :]
        else:
            anchor_img = anchor_imgs[anchor_idx, 0, 50, :]
            positive_img = positive_imgs[positive_idx, 0, 50, :]
            negative_img = anchor_imgs[negative_idx, 0, 50, :]

        axs[triplet_idx, 0].imshow(rerange_image(anchor_img))
        axs[triplet_idx, 0].set_title("Anchor")
        axs[triplet_idx, 0].axis("off")
        axs[triplet_idx, 1].imshow(rerange_image(positive_img))
        axs[triplet_idx, 1].set_title("Positive")
        axs[triplet_idx, 1].axis("off")
        axs[triplet_idx, 2].imshow(rerange_image(negative_img))
        axs[triplet_idx, 2].set_title("Negative")
        axs[triplet_idx, 2].axis("off")

    plt.tight_layout()
    plt.savefig(f"{result_path}/triplets_epoch_{epoch}.png")
    plt.close()
