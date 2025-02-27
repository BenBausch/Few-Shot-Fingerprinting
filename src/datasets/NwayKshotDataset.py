import random

import torch

from datasets.BaseDataset import BaseDataset


class NwayKshotDataset(BaseDataset):
    """
    This dataset class is used to create a N-way K-shot dataset.

    Args:
        dataset_path (str): Path to the dataset.
        split_file_path (str): Path to the split file.
        n_way (int): Number of classes.
        k_shot (int): Number of samples per class.
        q_queries (int): Number of queries per class.
        iterations (int): Number of iterations.
        file_endings (str): File endings of the images.
        anchor_modalities (list): List of anchor modalities.
        positive_modalities (list): List of positive modalities.
        anchor_transform (torchvision.transforms): Transformations for the anchor images.
        positive_transform (torchvision.transforms): Transformations for the positive images.
    """

    def __init__(
        self,
        dataset_name,
        dataset_path,
        split_file_path,
        meta_data_path,
        n_way,
        k_shot,
        k_modalities,
        q_queries,
        q_modalities,
        iterations,
        file_endings,
        min_images_per_modality=None,
        support_transform=None,
        query_transform=None,
    ):

        assert k_modalities is not None, "K modalities must be provided."
        assert q_modalities is not None, "Q modalities must be provided."

        # the modalities for the query and support set should be the same or have no intersection
        # e.g outof domain testing: the modalities should be different
        # e.g in domain testing: the modalities should be the same
        self.same_modalities = k_modalities == q_modalities
        assert self.same_modalities or (
            len(set(k_modalities).intersection(set(q_modalities))) == 0
        ), "The modalities for the query and support set should be the same or have no intersection."

        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.iterations = iterations
        self.k_modalities = k_modalities
        self.q_modalities = q_modalities
        self.support_transform = support_transform
        self.query_transform = query_transform

        super().__init__(
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            split_file_path=split_file_path,
            meta_data_path=meta_data_path,
            file_endings=file_endings,
            min_images_per_modality=min_images_per_modality,
        )

    def __len__(self):
        return self.iterations

    def get_n_subjects_randomly(self):
        """
        Get n subjects randomly.
        """
        # get n random subjects without replacement
        return random.sample(population=self.data, k=self.n_way)

    def get_k_shot_and_q_queries(self, subjects):
        """
        Get k-shot and q-queries for the subjects
        """
        queries = []
        supports = []

        for subject in subjects:
            # get all possible images with meta_data for the subjects support set
            possible_k_images = []
            for modality in self.k_modalities:
                for idx, i_data in enumerate(subject[modality]):
                    if self.meta_data is not None:
                        i_meta = {modality: subject["meta_data"][modality][idx]}
                    else:
                        i_meta = None
                    possible_k_images.append({"data": i_data, "meta": i_meta})

            if self.same_modalities:
                # the modalities for the query and support set are the same -->
                #   sample without replacement from the same set of images
                total_samples = self.k_shot + self.q_queries

                samples = random.sample(possible_k_images, total_samples)

                supports.append(samples[: self.k_shot])
                queries.append(samples[self.k_shot :])

            else:
                # the modalities for the query and support set are different -->
                #   sample without replacement from different sets of images
                k_samples = random.sample(possible_k_images, self.k_shot)
                supports.append(k_samples)

                # get all possible images for the subject query set
                possible_q_images = []
                for modality in self.q_modalities:
                    possible_q_images.extend(subject[modality])

                q_samples = random.sample(possible_q_images, self.q_queries)
                queries.append(q_samples)

        return supports, queries

    def __getitem__(self, idx):
        """
        Get the N-way K-shot data
        """
        self.current_batch = idx

        subjects = self.get_n_subjects_randomly()
        supports, queries = self.get_k_shot_and_q_queries(subjects)
        support_ids = []
        query_ids = []
        support_imgs = []
        support_meta = []
        query_imgs = []
        support_paths = []
        query_paths = []
        query_meta = []

        for i in range(self.n_way):
            # n subjects
            for j in range(self.k_shot):
                # k images per subject
                s_img = self.load_image_data(supports[i][j]["data"])

                if self.support_transform is not None:
                    s_img = self.support_transform(s_img)

                support_imgs.append(s_img)
                support_ids.append(subjects[i]["id"])
                support_paths.append(supports[i][j]["data"])
                support_meta.append(supports[i][j]["meta"])

            for j in range(self.q_queries):
                # q queries per subject
                q_img = self.load_image_data(queries[i][j]["data"])

                if self.query_transform is not None:
                    q_img = self.query_transform(q_img)

                query_imgs.append(q_img)
                query_ids.append(subjects[i]["id"])
                query_paths.append(queries[i][j]["data"])
                query_meta.append(queries[i][j]["meta"])

        meta_data = {
            "support_labels": support_ids,
            "query_labels": query_ids,
            "support_paths": support_paths,
            "query_paths": query_paths,
            "support_meta_data": support_meta if self.meta_data is not None else False,
            "query_meta_data": query_meta if self.meta_data is not None else False,
        }

        # Transform the list of tensors to a single tensor
        support_tensor = torch.stack(support_imgs)  # [n_way * k_shot, 1, H, W]
        query_tensor = torch.stack(query_imgs)  # [n_way * q_queries, 1, H, W]

        return support_tensor, query_tensor, meta_data
