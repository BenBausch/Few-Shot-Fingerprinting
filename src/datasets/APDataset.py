import random

from torch import Tensor

from datasets.BaseDataset import BaseDataset


class APDataset(BaseDataset):
    """
    Dataset class for subject data.

    Args:
        dataset_path (str): Path to the dataset.
        split_file_path (str): Path to the split file.
        return_images_as_y (bool): If True, the dataset will return the images as y.
        file_endings (str): File endings of the images.
        anchor_modalities (list): List of anchor modalities.
        positive_modalities (list): List of positive modalities.
        anchor_transform (torchvision.transforms): Transformations for the anchor images.
        positive_transform (torchvision.transforms): Transformations for the positive images.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.

    """

    def __init__(
        self,
        dataset_name,
        dataset_path,
        split_file_path,
        meta_data_path,
        file_endings=".nii.gz",
        anchor_modalities=["t1"],
        positive_modalities=None,
        anchor_transform=None,
        positive_transform=None,
        num_positives=1,
        min_images_per_modality=None,
    ):
        self.num_positives = num_positives

        assert anchor_modalities is not None, "Anchor modalities must be provided."
        self.anchor_modalities = anchor_modalities
        self.positive_modalities = positive_modalities
        self.anchor_transform = anchor_transform
        self.positive_transform = positive_transform

        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split_file_path=split_file_path,
            meta_data_path=meta_data_path,
            file_endings=file_endings,
            min_images_per_modality=min_images_per_modality,
        )

    def get_unique_random_if_possible(self, r_list, exclude_list=None):
        """
        Get a random element from a list, excluding the elements in the exclude list if possible.
        """
        if exclude_list is not None:
            r_list_diff = list(set(r_list).difference(set(exclude_list)))
            if len(r_list_diff) == 0:
                # no unique element left that is not in the exclude list
                return random.choice(r_list)
            else:
                return random.choice(r_list_diff)
        return random.choice(r_list)

    def get_random_modality(self, modalities, modality_exclude=None):
        return self.get_unique_random_if_possible(
            modalities, exclude_list=modality_exclude
        )

    def get_random_time_series_index(
        self, subject, modality, indexes_exclude=None
    ) -> int:
        time_series = subject[modality]
        return self.get_unique_random_if_possible(
            range(len(time_series)), exclude_list=indexes_exclude
        )

    def load_subject_modality_at_index(
        self, subject, modality, time_series_index=0
    ) -> Tensor:
        img = self.load_image_data(subject[modality][time_series_index])
        return img

    def __getitem__(self, idx):

        # get the anchor image
        anchor_subject = self.data[idx]
        anchor_modality = self.get_random_modality(self.anchor_modalities)
        anchor_time_series_index = self.get_random_time_series_index(
            anchor_subject, anchor_modality
        )
        anchor = self.load_subject_modality_at_index(
            subject=anchor_subject,
            modality=anchor_modality,
            time_series_index=anchor_time_series_index,
        )

        if self.anchor_transform is not None:
            anchor = self.anchor_transform(anchor)

        data = {"anchor_img": anchor}
        meta_data = {
            "anchor_id": anchor_subject["id"],
            "anchor_modality": anchor_modality,
            "anchor_time_series_index": anchor_time_series_index,
        }

        # get the positive image
        indexes_to_ignore = {anchor_time_series_index}
        modalities_to_ignore = {anchor_modality}

        for n in range(self.num_positives):
            positive_sample = None
            if self.positive_modalities is not None:
                positive_modality = self.get_random_modality(
                    self.positive_modalities,
                    modality_exclude=list(modalities_to_ignore),
                )
                modalities_to_ignore.add(positive_modality)
                positive_ts_idx = self.get_random_time_series_index(
                    anchor_subject,
                    positive_modality,
                    indexes_exclude=list(indexes_to_ignore),
                )
                indexes_to_ignore.add(positive_ts_idx)
                positive_sample = self.load_subject_modality_at_index(
                    subject=anchor_subject,
                    modality=positive_modality,
                    time_series_index=positive_ts_idx,
                )

                if self.positive_transform is not None:
                    positive_sample = self.positive_transform(positive_sample)

                img_key = f"positive_img_{n}"
                modality_key = f"positive_modality_{n}"
                ts_key = f"positive_time_series_index_{n}"

                data[img_key] = positive_sample
                meta_data[modality_key] = positive_modality
                meta_data[ts_key] = positive_ts_idx

        return data, meta_data
