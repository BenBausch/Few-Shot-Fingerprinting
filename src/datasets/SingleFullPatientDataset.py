from datasets.BaseDataset import BaseDataset


class SingleFullSubjectDataset(BaseDataset):
    """
    Loads the data for a single full subject.
    """

    def __init__(
        self,
        dataset_name,
        dataset_path,
        split_file_path,
        meta_data_path,
        file_endings=".nii.gz",
        min_images_per_modality=None,
        transforms=None,
    ):
        self.transforms = transforms
        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split_file_path=split_file_path,
            meta_data_path=meta_data_path,
            file_endings=file_endings,
            min_images_per_modality=min_images_per_modality,
        )

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, index):
        subject = self.data[index]

        # Load all the images for each modality
        data = {
            "images": [],
            "subject_id": subject["id"],
            "paths": [],
            "meta_data": [],
            "modalities": [],
        }

        for modality in subject["modalities"]:
            for idx, img_path in enumerate(subject[modality]):
                # load the image and apply test time transformations as they do not augment the data
                img_data = self.load_image_data(img_path)
                if self.transforms is not None:
                    img_data = self.transforms(img_data)
                data["images"].append(img_data)
                data["paths"].append(img_path)
                data["modalities"].append(modality)
                if self.meta_data is not None:
                    data["meta_data"].append(subject["meta_data"][modality][idx])

        return data
