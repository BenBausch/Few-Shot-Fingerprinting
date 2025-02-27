import os
import pickle

import numpy as np
import torch
from monai.networks.nets import ViT
from tqdm import tqdm

from plotting.cluster import plot_cluster_tsne


class Embedder:
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
        max_batch_size: int = 20,
    ):
        self.config = config
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.result_path = result_path
        self.max_batch_size = max_batch_size

    def embed_data(self):
        """
        Embeds the data using the given model and saves the embeddings to the result_path.
        """
        self.model.eval()
        embeddings = {}

        with torch.no_grad():
            for subject in tqdm(self.data_loader, total=len(self.data_loader)):

                embeddings[subject["subject_id"][0]] = {}
                # save all the data except the images
                for key, value in subject.items():
                    if key != "images" and (key == "meta_data" and value != []):
                        embeddings[subject["subject_id"][0]][key] = value[0]

                # Embed the images
                images = torch.cat(subject["images"], dim=0)
                tensor_list = images.split(self.max_batch_size, dim=0)
                embedding_list = []
                for tensor in tensor_list:
                    tensor = tensor.to(self.device)
                    # This is for the ViT model where it outputs a tuple.
                    embedding = self.model(tensor)

                    if isinstance(self.model, ViT):
                        embedding = embedding[0]

                    embedding_list.append(embedding)

                embeddings[subject["subject_id"][0]]["embeddings"] = (
                    torch.cat(embedding_list, dim=0).detach().cpu().numpy().tolist()
                )

        os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
        # save the pickled dictionary with numpy embeddings
        with open(self.result_path, "wb") as f:
            pickle.dump(embeddings, f)

    @staticmethod
    def plot_embedded_data(pickle_path):
        """
        Plots the embedded data using TSNE and saves the plot to the result_path.
        """
        with open(pickle_path, "rb") as f:
            embedding_dict = pickle.load(f)

        # Create the labels for each single image and extract the embeddings
        labels = []
        embeddings = []
        for k, v in embedding_dict.items():
            labels.extend([k] * len(embedding_dict[k]["embeddings"]))
            embeddings.extend(embedding_dict[k]["embeddings"])
        # Convert the embeddings to a numpy array
        embeddings = np.array(embeddings)

        # Plot the embeddings
        plot_result_path = os.path.join(os.path.dirname(pickle_path), "tsne.svg")
        plot_cluster_tsne(
            embeddings=embeddings,
            labels=labels,
            dims=2,
            perplexity=20,
            result_path=plot_result_path,
            animated=False,
        )