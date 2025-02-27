import itertools
import logging
import os
from typing import List

import numpy as np
import torch
from monai.networks.nets import ViT
from tqdm import tqdm

import wandb
from metrics.retrieval import (
    get_ranked_retrieval_matrix,
    get_relevancy_matrix,
    mean_hit_at_k,
    mean_majority_vote_hit_at_k,
    mean_recall_at_k,
    order_by_similarities,
)
from plotting.similarity_matrix import plot_similiarity_matrix
from plotting.support_query import plot_query_support
from plotting.tsne import plot_tsne
from similarity_fn.QuerySupportSimilarity import QuerySupportSimilarity


class Tester:
    __test__ = False  # This class is not a pytest test

    def __init__(
        self,
        config: dict,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        result_path: str,
        n_way: int = None,
        k_shot: int = None,
        q_queries: int = None,
        top_k: int = 5,
        max_batch_size: int = 20,
        similarity_metric: str = "euclidean",
        is_val_tester: bool = False,
        wandb_log: bool = False,
    ):
        """
        Initialize the Tester class.

        Args:
            config (dict): Configuration dictionary
            model (torch.nn.Module): Model to be tested
            test_loader (torch.utils.data.DataLoader): DataLoader for testing
            device (torch.device): Device to run the model on
            result_path (str): Path to save the results
            n_way (int): Number of classes in a task
            k_shot (int): Number of support examples per class
            log_frequency (int): Log frequency
            max_batch_size (int): Maximum batch size
        """
        self.config = config
        if not is_val_tester:
            self.plotting_config = config["PLOTTING_TESTING"]
        else:
            self.plotting_config = config["PLOTTING_VALIDATION"]
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.result_path = result_path
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.top_k = top_k
        self.num_supports = n_way * k_shot
        self.max_batch_size = max_batch_size
        self.wandb_log = wandb_log
        self.sim_fn = QuerySupportSimilarity(similarity_metric=similarity_metric)
        self.is_val_tester = is_val_tester
        os.makedirs(self.result_path, exist_ok=True)

        self.model.to(self.device)

    def get_embedding_list_from_tensor(self, tensor: torch.Tensor):
        """
        Get the list of embeddings from the tensor.

        Args:
            tensor (torch.Tensor): Tensor of images.
                Shape: (batch_size, batch_dimension, channels, (img dimensions)),
                    where batch_dimension is n_way * k_shot or n_way * q_queries
            example: (1, 20, 1, 128, 128) for 2D images, (1, 20, 1, 32, 32, 32) for 3D images
        """
        with torch.no_grad():
            # Split the tensor into list of tensors that have a batch size of max_batch_size
            tensor_list = tensor[0].split(self.max_batch_size, dim=0)
            embedding_list = []
            for tensor in tensor_list:
                tensor = tensor.to(self.device)
                embedding = self.model(tensor)
                if isinstance(self.model, ViT):
                    embedding = embedding[0]
                embedding_list.append(embedding)
            return torch.cat(embedding_list, dim=0)

    def calculate_similarity(
        self, support_embedding: torch.Tensor, query_embedding: torch.Tensor
    ):
        """
        Calculate the similarity between the support and query set

        Args:
            support_embedding (torch.Tensor): Embedding of the support set
            query_embedding (torch.Tensor): Embedding of the query set

        Returns:
            torch.Tensor: Similarity between the support and query set of shape (n_way * q_queries, n_way * k_shot)
        """
        # Calculate the cosine similarity
        similarity = self.sim_fn(
            query_embedding=query_embedding,
            support_embedding=support_embedding,
        )

        return similarity

    def calculate_metrics(
        self,
        similarity: torch.Tensor,
        query_labels: List[List],
        support_labels: List[List],
    ):
        """
        Calculate the metrics for the model

        Args:
            similarity (torch.Tensor): Similarity between the support and query set
            query_labels (list): List of query labels
            support_labels (list): List of support labels
        """
        # Get the unique_labels and map them to integers for numpy operations
        flatten_query_labels = list(itertools.chain.from_iterable(query_labels))
        flatten_support_labels = list(itertools.chain.from_iterable(support_labels))
        unique_query_labels = set(flatten_query_labels)
        unique_support_labels = set(flatten_support_labels)
        unique_labels = list(unique_query_labels.union(unique_support_labels))
        labels_to_int = {label: i for i, label in enumerate(unique_labels)}
        support_labels = np.array(list(map(labels_to_int.get, flatten_support_labels)))
        query_labels = np.array(list(map(labels_to_int.get, flatten_query_labels)))

        # get the binary matrix indicating the relevancy of each support image to each query image
        relevancy_matrix = get_relevancy_matrix(
            query_labels=query_labels, support_labels=support_labels
        )
        # order the relevancy matrix by decreasing similarity
        ordered_rel = order_by_similarities(
            array=relevancy_matrix, similarities=similarity
        )

        # Calculate the recall at k for each query,
        # k is equal to the number of supports per class
        # if k < k_shot, then recall can not be 100%
        # if k > k_shot, then recall can be 100% with a higher probability then the prior probability
        mean_recall_at_k_shot = mean_recall_at_k(k=self.k_shot, ordered_rel=ordered_rel)

        mean_hit_at_i = {}
        for i in range(1, self.top_k + 1):
            # Calculate the hit rate at k for each query,
            # k is the allowed top number of supports to be considered
            # the bigger k, the higher the possibility of having a higher hit rate
            # at K = k_shot * n_way, the hit rate is 100%
            mean_hit_at_i[i] = mean_hit_at_k(k=i, ordered_rel=ordered_rel)

        # Calculate the hit rate at k for each query using majority vote,
        # k is the equal to the number of supports per class
        # if k < k_shot, then hit rate can still be 100% but with a lower
        # probability than the prior probability¨
        # if k > k_shot, then hit rate can be 100% with a higher probability
        # then the prior probability
        # this metric is measures the performance of retrieval of the subject instead of
        # single samples from the support set

        # Calculate the ranked labels per query
        ranked_labels_per_query = get_ranked_retrieval_matrix(
            support_labels=support_labels,
            similarities=similarity,
        )
        mean_majority_vote_hit_at_k_shot = mean_majority_vote_hit_at_k(
            k=self.k_shot,
            relevant_labels=query_labels,
            labels=ranked_labels_per_query,
        )

        return {
            "mean_recall_at_k_shot": mean_recall_at_k_shot,
            "mean_hit_at_i": mean_hit_at_i,
            "mean_majority_vote_hit_at_k_shot": mean_majority_vote_hit_at_k_shot,
        }

    def test(self):
        """
        Test the model
        """
        self.model.eval()

        logging.info(
            f"Testing the model in {self.n_way}-way {self.k_shot}-shot setting"
        )
        mean_metrics = {
            "mean_recall_at_k_shot": 0,
            "mean_hit_at_i": {k: 0 for k in range(1, self.top_k + 1)},
            "mean_majority_vote_hit_at_k_shot": 0,
        }

        for idx, (support_set, query_set, meta_data) in tqdm(
            enumerate(self.test_loader)
        ):
            # Embedding of the support and query set
            support_embedding = self.get_embedding_list_from_tensor(support_set)
            query_embedding = self.get_embedding_list_from_tensor(query_set)

            # Calculate the similarity between the support and query set embeddings
            similarity = self.calculate_similarity(
                support_embedding=support_embedding, query_embedding=query_embedding
            )

            # Calculate the metrics
            idx_metrics = self.calculate_metrics(
                similarity=similarity,
                query_labels=meta_data["query_labels"],
                support_labels=meta_data["support_labels"],
            )

            # Update the mean metrics
            for key in mean_metrics.keys():
                if isinstance(mean_metrics[key], dict):
                    for k in mean_metrics[key].keys():
                        mean_metrics[key][k] += idx_metrics[key][k]
                else:
                    mean_metrics[key] += idx_metrics[key]

            # Plot the results
            if idx % self.plotting_config["FREQUENCY"] == 0:
                logging.info(f"Batch: {idx}/{len(self.test_loader)}")

                links = {}
                # support set is of shape [n_way * k_shot, 1, H, W]
                # query set is of shape [n_way * q_queries, 1, H, W]
                for i in range(0, self.n_way):
                    # for each index of image in the support set, create a link
                    # to the corresponding image indexes in the query set
                    # example: {0: [0, 1, 2], 1: [3, 4, 5]} supposing n_way=2, k_shot=1, q_queries=3
                    for k_s in range(self.k_shot):
                        anchor_index = i * self.k_shot + k_s
                        for q in range(self.q_queries):
                            query_index = i * self.q_queries + q
                            if anchor_index not in links:
                                links[anchor_index] = [query_index]
                            else:
                                links[anchor_index].append(query_index)

                if self.plotting_config["3D_TSNE"]["PLOT"]:
                    plot_tsne(
                        result_path=self.result_path,
                        anchor_embeddings=support_embedding.detach().cpu().numpy(),
                        positive_embeddings=query_embedding.detach().cpu().numpy(),
                        epoch=idx,
                        links=links,
                        dims=3,
                        group_by=self.config["PLOTTING_TESTING"]["3D_TSNE"]["GROUP_BY"],
                        anchor_meta_data=meta_data["support_meta_data"],
                        positive_meta_data=meta_data["query_meta_data"],
                    )

                if self.plotting_config["SUPPORT_QUERY"]["PLOT"]:
                    plot_query_support(
                        result_path=self.result_path,
                        support_set=support_set,
                        query_set=query_set,
                        meta_data=meta_data,
                        epoch=idx,
                        k_shot=self.k_shot,
                        n_way=self.n_way,
                        q_queries=self.q_queries,
                    )

                if self.plotting_config["SIMILARITY_MATRIX"]["PLOT"]:
                    plot_similiarity_matrix(
                        matrix=similarity,
                        result_path=os.path.join(
                            self.result_path, f"similarity_matrix_{idx}.png"
                        ),
                        highlight_row=True,
                    )

        # Calculate the average metrics
        avg_recall_at_k_shot = mean_metrics["mean_recall_at_k_shot"] / len(
            self.test_loader
        )
        avg_topk_hit = {
            k: round(v / len(self.test_loader), 4)
            for k, v in mean_metrics["mean_hit_at_i"].items()
        }
        avg_majority_vote_hit_at_k_shot = mean_metrics[
            "mean_majority_vote_hit_at_k_shot"
        ] / len(self.test_loader)

        logging.info(f"Mean Recall at {self.k_shot}-shot: {avg_recall_at_k_shot:.4f}")
        logging.info(f"Mean Hit at {self.top_k} for each query: {avg_topk_hit}")
        logging.info(
            f"Mean Majority Vote Hit at {self.k_shot}-shot: {avg_majority_vote_hit_at_k_shot:.4f}"
        )

        if self.wandb_log:
            wandb.log(
                {
                    f"Mean Recall at K-shot {self.k_shot}": avg_recall_at_k_shot,
                    "Mean Hit at top k": avg_topk_hit,
                    f"Mean Majority Vote Hit at K-shot {self.k_shot}": avg_majority_vote_hit_at_k_shot,
                }
            )
        return avg_recall_at_k_shot, avg_topk_hit, avg_majority_vote_hit_at_k_shot
