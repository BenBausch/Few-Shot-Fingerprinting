import logging
import os

import nibabel as nib
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from datasets import create_mri_subject, create_omniglot_subject, create_xray_subject

CHESTXRAY14 = "chestxray14"
OMNIGLOT = "omniglot"
BRATS2021 = "brats2021"
SUPPORTED_DATASETS = [OMNIGLOT, CHESTXRAY14, BRATS2021]


class BaseDataset(Dataset):
    """
    Base dataset class that handels the subject creation and loading of the dataset.
    """

    def __init__(
        self,
        dataset_name,
        dataset_path,
        meta_data_path,
        split_file_path,
        file_endings=".png",
        min_images_per_modality=None,
    ):
        assert (
            dataset_name in SUPPORTED_DATASETS
        ), f"Dataset {dataset_name} is not supported."

        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.split_file_path = split_file_path
        self.file_endings = file_endings
        self.min_images_per_modality = min_images_per_modality

        self.set_image_loader()

        self.load_meta_data(meta_data_path)
        self.load_data()

    def load_meta_data(self, meta_data_path):
        if meta_data_path is not None:
            self.meta_data = self.meta_data = pd.read_csv(meta_data_path)
        else:
            self.meta_data = None

    def load_data(self):
        with open(self.split_file_path, "r") as f:
            self.subject_ids = f.read().splitlines()
        self.subject_paths = []

        # Load the subjects data and create tio.subjects
        self.data = []

        filtered_subject_ids = (
            []
        )  # These lists need to be here, or else you would be appending the ids over and over again.
        filtered_subject_paths = []

        for idx, subject_id in tqdm(
            enumerate(self.subject_ids),
            total=len(self.subject_ids),
            desc="Creating subjects",
        ):
            if self.dataset_name == OMNIGLOT:
                self.subject_paths.append(
                    os.path.join(self.dataset_path, *subject_id.split("/"))
                )
                subject = create_omniglot_subject(
                    self.subject_paths[idx], self.file_endings
                )

            elif self.dataset_name == CHESTXRAY14:
                self.subject_paths.append(os.path.join(self.dataset_path, subject_id))
                subject = create_xray_subject(
                    self.subject_paths[idx], self.file_endings, meta_data=self.meta_data
                )

            elif self.dataset_name == BRATS2021:
                self.subject_paths.append(os.path.join(self.dataset_path, subject_id))
                subject = create_mri_subject(self.subject_paths[idx], self.file_endings)

            # Check if the subject has the required number of images per modality
            ignore_subject = False
            for modality in subject["modalities"]:
                if (
                    self.min_images_per_modality is not None
                    and len(subject[modality]) < self.min_images_per_modality
                ):
                    ignore_subject = True
            if ignore_subject:
                continue

            self.data.append(subject)
            filtered_subject_ids.append(subject_id)
            filtered_subject_paths.append(self.subject_paths[idx])

        self.subject_ids = filtered_subject_ids
        self.subject_paths = filtered_subject_paths

        logging.info(f"Retained {len(self.subject_ids)} subjects after filtering.")

    def __len__(self):
        return len(self.subject_ids)

    def set_image_loader(self):
        if self.file_endings == ".png":
            self.load_image_data = self.png_loader
        elif self.file_endings == ".nii.gz":
            self.load_image_data = self.nii_loader
        else:
            raise ValueError(f"File endings {self.file_endings} is not supported.")

    def nii_loader(self, path):
        nifti_img = nib.load(path)
        image_data = nifti_img.get_fdata()
        tensor_data = torch.from_numpy(image_data).to(dtype=torch.float32)
        image = tensor_data.permute(2, 0, 1).unsqueeze(0)
        image = (image - image.min()) / (image.max() - image.min())
        return image

    def png_loader(self, path):
        image = Image.open(path)
        self.pil_to_tensor = transforms.PILToTensor()
        if self.dataset_name == CHESTXRAY14:
            image = image.convert("RGB")
            image = self.pil_to_tensor(image).to(dtype=torch.float32)
            image = image / 255.0
        else:
            image = self.pil_to_tensor(image).to(dtype=torch.float32)
        return image
