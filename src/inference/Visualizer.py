import os
import shutil

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm


def save_nifti(tensor, path):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    tensor = (tensor * 255).astype("uint8")

    img = nib.Nifti1Image(tensor, np.eye(4))
    nib.save(img, path)


def image_from_tensor(img, as_uint8=True):
    img = img.cpu().numpy()
    img = img - img.min()
    img = img / img.max()
    img = img.squeeze(0).transpose(1, 2, 0)
    if as_uint8:
        img = (img * 255).astype("uint8")
    return img


class SimilarityToConceptTarget:
    def __init__(self, features):
        self.features = features

    def __call__(self, model_output):
        cos = torch.nn.CosineSimilarity(dim=0)
        return cos(model_output, self.features)
        # similarity version 2
        # d = torch.nn.PairwiseDistance(p=2)
        # return torch.exp(-d(model_output, self.features))
        # similarity version 3
        # d = torch.nn.PairwiseDistance(p=2)
        # dist = d(model_output, self.features)
        # return 1 / (1 + dist)


class Visualizer:
    """
    This class is responsible for embedding the data using the given model.
    It will embed the data and save the embeddings to the result_path.
    """

    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        result_path: str,
        max_batch_size: int = 1,
    ):
        self.config = config
        self.model = model
        if "ResNet50" in config["MODEL"]["NAME"]:
            self.target_layers = [self.model.layer4]

        self.data_loader = data_loader
        self.device = device
        self.result_path = result_path
        self.max_batch_size = max_batch_size

    def visualize_data(self):
        if self.config["DATASET"]["DATASET_NAME"] == "brats2021":
            self.gradcam_3d()
        elif self.config["DATASET"]["DATASET_NAME"] == "chestxray14":
            self.gradcam_2d()

    def gradcam_2d(self):
        """
        Embeds the data using the given model and saves the embeddings to the result_path.
        """
        self.model.eval()
        shutil.rmtree(self.result_path, ignore_errors=True)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)

        for subject in tqdm(self.data_loader, total=len(self.data_loader)):
            if len(subject["images"]) >= 2:
                s_path = os.path.join(self.result_path, subject["subject_id"][0])
                os.makedirs(s_path, exist_ok=True)

                # Create a figure to hold all images
                n_images = len(subject["images"])
                fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))

                first_img = subject["images"][0].to(self.device)
                first_embedding = self.model(first_img)[0, :]
                axes[0].imshow(image_from_tensor(first_img, as_uint8=True))
                axes[0].axis("off")
                axes[0].set_title("Original")

                for idx, img in enumerate(subject["images"]):
                    img = img.to(self.device)
                    similarity = [SimilarityToConceptTarget(first_embedding)]
                    with GradCAM(
                        model=self.model, target_layers=self.target_layers
                    ) as cam:
                        grayscale_cam = cam(input_tensor=img, targets=similarity)[0, :]
                        image = image_from_tensor(img, as_uint8=False)
                        grad_cam_image = show_cam_on_image(
                            image, grayscale_cam, use_rgb=True
                        )
                        embed = self.model(img)[0, :]
                        distance = similarity[0](embed).item()

                        axes[idx].imshow(grad_cam_image)
                        axes[idx].axis("off")
                        axes[idx].set_title("distance: {:.4f}".format(distance))

                plt.tight_layout()
                plt.savefig(os.path.join(s_path, "comparison.jpg"))
                plt.close(fig)

    def gradcam_3d(self):
        """
        Embeds the data using the given model and saves the embeddings to the result_path.
        """
        self.model.eval()
        shutil.rmtree(self.result_path, ignore_errors=True)
        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)

        for subject in tqdm(self.data_loader, total=len(self.data_loader)):
            if len(subject["images"]) >= 2:
                s_path = os.path.join(self.result_path, subject["subject_id"][0])
                os.makedirs(s_path, exist_ok=True)

                first_img = subject["images"][0].to(self.device)
                print(subject["modalities"][0])
                first_embedding = self.model(first_img)[0, :]

                for idx, (img, modal) in enumerate(
                    zip(subject["images"], subject["modalities"])
                ):
                    img = img.to(self.device)
                    similarity = [SimilarityToConceptTarget(first_embedding)]
                    with GradCAM(
                        model=self.model, target_layers=self.target_layers
                    ) as cam:
                        grayscale_cam = cam(input_tensor=img, targets=similarity)[0, :]

                        # save as nifti
                        nifti_path = os.path.join(s_path, f"cam_{modal}.nii.gz")
                        save_nifti(grayscale_cam, nifti_path)

                        # save the original image
                        orig_path = os.path.join(s_path, f"orig_{modal}.nii.gz")
                        save_nifti(img[0, 0, :], orig_path)
