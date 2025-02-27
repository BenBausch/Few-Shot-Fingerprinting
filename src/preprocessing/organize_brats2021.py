import glob
import os
import random
from typing import List, Tuple
from pathlib import Path


def organize_brats2021_folder_structure(path, savepath):

    files = glob.glob(path + "*")

    filenames_for_dirs = []
    for file in files:
        filename = file.split(".")[0].split("_")[1]
        filenames_for_dirs.append(filename)

    filenames_for_dirs = sorted(list(set(filenames_for_dirs)))

    for filename in filenames_for_dirs:
        os.makedirs(savepath + filename, exist_ok=True)

        # get the images in the files that correspond to this filename
        images = glob.glob(path + "*" + filename + "*")

        # move each image to the new directory for this filename
        for image in images:
            os.rename(image, savepath + filename + "/" + image.split("/")[-1])

    print("Dataset has been reorganized!")


def split_folders(
    folder_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    output_dir: str = "splits",
    random_seed: int = 42,
):
    """
    Split a list of folders into train, validation, and test sets and save them as text files.

    Args:
        folder_list: List of folder paths to split
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        output_dir: Directory to save the split files (default: "splits")
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple containing the train, validation, and test splits as lists
    """
    # Get list of folders
    foldes = glob.glob(folder_dir + "*")
    folder_list = [folder.split("/")[-1] for folder in foldes]

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-10:
        raise ValueError("Split ratios must sum to 1")

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle the folder list
    shuffled_folders = folder_list.copy()
    random.shuffle(shuffled_folders)

    # Calculate split indices
    total_samples = len(shuffled_folders)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    # Split the data
    train_split = shuffled_folders[:train_size]
    val_split = shuffled_folders[train_size : train_size + val_size]
    test_split = shuffled_folders[train_size + val_size :]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save splits to text files
    def save_split(split: List[str], filename: str):
        filepath = Path(output_dir) / filename
        with open(filepath, "w") as f:
            f.write("\n".join(split))

    save_split(train_split, "train.txt")
    save_split(val_split, "val.txt")
    save_split(test_split, "test.txt")

    print(f"Data split successfully saved to {output_dir}")


if __name__ == "__main__":
    split_folders(
        folder_dir="/path/to/brats2021_data/",
        output_dir="/path/to/brats2021_splits/",
    )
