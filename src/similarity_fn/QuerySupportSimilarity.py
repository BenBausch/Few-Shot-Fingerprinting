import numpy as np
import torch


class QuerySupportSimilarity:
    def __init__(self, similarity_metric="p_norm", p=2.0):

        self.similarity_metric = similarity_metric
        self.p = p
        self._get_similarity_metric()

    def _get_similarity_metric(self):
        """
        Get the similarity metric
        """
        if self.similarity_metric == "p_norm":
            self.sim_fn = torch.nn.PairwiseDistance(p=self.p, eps=1e-06, keepdim=False)
        elif self.similarity_metric == "cosine":
            cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
            self.sim_fn = lambda x, y: (1 - cosine_sim(x, y)) / 2
        else:
            raise ValueError(
                f"Similarity metric {self.similarity_metric} is not supported."
            )

    def forward(self, support_embedding: torch.Tensor, query_embedding: torch.Tensor):
        """
        Calculate the similarity between the support and query set

        Args:
            query_embedding (torch.Tensor): Embedding of the support set. Shape: (n_query, embedding_dim)
            support_embedding (torch.Tensor): Embedding of the query set. Shape: (n_support, embedding_dim)
        """
        # Initialize the similarity matrix shape (n_query, n_support)
        sim_matrix = np.zeros(
            shape=(query_embedding.shape[0], support_embedding.shape[0])
        )
        # Calculate the cosine similarity of the query embedding to each support embedding
        for i, q_embed in enumerate(query_embedding.split(1, dim=0)):
            # Calculate the cosine similarity between the query embedding and the single embedding out of the support set
            similarity = self.sim_fn(
                q_embed.repeat(support_embedding.shape[0], 1), support_embedding
            )
            # Detach the tensor and convert it to a numpy array
            similarity = similarity.detach().cpu().numpy()
            sim_matrix[i, :] = similarity

        if self.similarity_metric == "euclidean":
            sim_matrix = 1 - (sim_matrix / np.max(sim_matrix, axis=1, keepdims=True))

        return sim_matrix

    def __call__(self, support_embedding: torch.Tensor, query_embedding: torch.Tensor):
        return self.forward(support_embedding, query_embedding)

