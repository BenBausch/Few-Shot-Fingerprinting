import random
from typing import Literal

import numpy as np
import torch

allowed_loss_types = ["triplet_margin_loss", "exponential_distance_loss"]


class CustomTripletMarginLoss(torch.nn.Module):
    def __init__(
        self,
        loss_type="triplet_margin_loss",
        margin=1,
        ratio_hard_easy_negative=0.8,
        p_norm: int = 2,
        distance_metric="p_norm",
    ):
        super(CustomTripletMarginLoss, self).__init__()

        self.margin = margin
        self.ratio_hard_easy_negative = ratio_hard_easy_negative
        self.p_norm = p_norm
        self.eps = 1e-6

        if distance_metric == "p_norm":
            self.dist_fn = torch.nn.PairwiseDistance(p=p_norm, eps=self.eps)
        elif distance_metric == "cosine":
            # get cosine_sim
            cosine_sim = torch.nn.CosineSimilarity(dim=1, eps=self.eps)
            # convert to distances
            self.dist_fn = lambda x, y: (1 - (cosine_sim(x, y))) / 2
        else:
            raise ValueError(
                f"Distance metric {distance_metric} is not supported. Supported metrics are ['p_norm', 'cosine']"
            )

        self.loss_type = loss_type
        if self.loss_type not in allowed_loss_types:
            raise ValueError(
                f"loss_type should be one of {allowed_loss_types}, got {self.loss_type}"
            )
        elif self.loss_type == "triplet_margin_loss":
            self.loss_fn = self.triplet_margin_loss
        elif self.loss_type == "exponential_distance_loss":
            self.loss_fn = self.exponential_distance_loss

    def forward(
        self, embeddings, p_embeddings, embedding_ids, return_triplet_indexes=False
    ):
        """
        Calculate the triplet loss for the embeddings and positive embeddings

        Args:
            embeddings (torch.Tensor): The embeddings of the anchor batch
            p_embeddings (torch.Tensor): The embeddings of the positives batch
            embedding_ids (list[str]): The ids of the embeddings
        """

        loss = self.semi_hard_loss(
            dist_fn=self.dist_fn,
            margin=self.margin,
            loss_fn=self.loss_fn,
            embeddings=embeddings,
            p_embeddings=p_embeddings,
            embedding_ids=embedding_ids,
            ratio_hard_easy_negative=self.ratio_hard_easy_negative,
            return_triplet_indexes=return_triplet_indexes,
        )

        if return_triplet_indexes:
            return loss
        else:
            return loss[0]

    def all_losses(self, embeddings, p_embeddings, embedding_ids):
        """
        Calculate the triplet loss for the embeddings and positive embeddings

        Args:
            embeddings (torch.Tensor): The embeddings of the anchor batch
            p_embeddings (torch.Tensor): The embeddings of the positives batch
            embedding_ids (list[str]): The ids of the embeddings

        Returns:
            torch.Tensor: The loss value
            torch.Tensor: The positive distances
            torch.Tensor: The negative distances
        """
        return self.semi_hard_loss(
            dist_fn=self.dist_fn,
            margin=self.margin,
            loss_fn=self.loss_fn,
            embeddings=embeddings,
            p_embeddings=p_embeddings,
            embedding_ids=embedding_ids,
        )

    @staticmethod
    def triplet_margin_loss(postive_distance, negative_distance, margin):
        triplet_loss = torch.clamp(postive_distance - negative_distance + margin, min=0)
        return (
            torch.mean(triplet_loss),
            torch.mean(postive_distance),
            torch.mean(negative_distance),
        )

    @staticmethod
    def exponential_distance_loss(postive_distance, negative_distance, *args, **kwargs):
        neg_loss = torch.exp(-negative_distance)
        pos_loss = 1 - torch.exp(-postive_distance)
        return (
            torch.mean(pos_loss) + torch.mean(neg_loss),
            torch.mean(pos_loss),
            torch.mean(neg_loss),
        )

    @staticmethod
    def _get_true_negative_indexes(embedding_ids: list[str]):
        """
        Returns a list of indexes of the embeddings that are not of the same id as the anchor embedding.

        Args:
            embedding_ids (list[str]): List of the subject ids of the embeddings
        """
        per_anchor_negative_indexes = []
        for i, anchor_id in enumerate(embedding_ids):
            indexes = []
            for j, embedding_id in enumerate(embedding_ids):
                if embedding_id != anchor_id:
                    indexes.append(j)
            per_anchor_negative_indexes.append(indexes)
        return per_anchor_negative_indexes

    @staticmethod
    def mine_semi_hard_negative(
        a_embedding: torch.Tensor,
        negative_embeddings: torch.Tensor,
        positive_distance: torch.Tensor,
        dist_fn,
        margin: float,
        strictly_hard_negative: bool = False,
    ):
        with torch.no_grad():
            # prevent the calculations while mining the negative samples
            # from being part of the computation graph
            anchor = a_embedding.repeat(negative_embeddings.shape[0], 1)
            distances = dist_fn(anchor, negative_embeddings)
            # get all the indexes of negatives that are semi-hard
            if strictly_hard_negative:
                all_semi_hard_indexes = torch.where(
                    (distances - positive_distance.detach()) < 0
                )[0].tolist()
            else:
                all_semi_hard_indexes = torch.where(
                    (distances - positive_distance.detach()) < margin
                )[0].tolist()

            if all_semi_hard_indexes == []:
                # if no semi-hard negative samples are found, return the hardest negative sample.
                # Postive_distance is a constant and does not influence the result.
                all_semi_hard_indexes = [torch.argmin(distances).item()]

            # choose a random semi-hard negative sample
            chosen_index = (
                np.random.choice(all_semi_hard_indexes)
                if len(all_semi_hard_indexes) > 0
                else None
            )
            return chosen_index

    @staticmethod
    def semi_hard_loss(
        dist_fn,
        margin,
        loss_fn,
        embeddings: torch.Tensor,
        p_embeddings: torch.Tensor,
        embedding_ids: list[str],
        ratio_hard_easy_negative: float,
        return_triplet_indexes: bool = False,
    ) -> tuple[Literal[0], Literal[0], Literal[0]]:
        """
        Select for each anchor positive pair a semi hard example where the difference between the
        anchor-positive distance and the anchor-negative distance is less than margin.

        Args:
            embeddings (torch.Tensor): The embeddings of the anchor batch
            p_embeddings (torch.Tensor): The embeddings of the positives batch
            margin (float): The margin for the triplet loss
            embedding_ids (list[str]): The ids of the embeddings
            dist_fn: The distance function to calculate the distance between embeddings
            loss_fn: The loss function to calculate the loss
        """
        # iterate over all embedding ids,
        # to get the a random semi-hard negative sample of all other anchors for each anchor
        all_positive_distances = dist_fn(embeddings.detach(), p_embeddings.detach())
        triplet_indexes = []
        retained_negative_distances = []
        retained_positive_distances = []
        anchors_already_retained = set()
        # detach the embeddings to avoid backpropagation when selecting the negatives
        true_negative_indexes = CustomTripletMarginLoss._get_true_negative_indexes(
            embedding_ids=embedding_ids
        )
        weight_negative = ratio_hard_easy_negative
        weight_positive = 1 - ratio_hard_easy_negative

        # calculate the distance between each anchor and all other embeddings of different ids
        for a_idx, tn_indexes in enumerate(true_negative_indexes):
            if (embedding_ids[a_idx] in anchors_already_retained) or (
                len(tn_indexes) == 0
            ):
                # if the anchor has already been retained or has no semi-hard-negatives
                continue
            else:
                # each anchor is retained only once as the retained negative sample might be the same for the same anchor
                pass
                # anchors_already_retained.add(embedding_ids[a_idx])

            chosen_index = CustomTripletMarginLoss.mine_semi_hard_negative(
                a_embedding=embeddings.detach()[a_idx].unsqueeze(0),
                negative_embeddings=embeddings.detach()[tn_indexes],
                positive_distance=all_positive_distances[a_idx],
                margin=margin,
                dist_fn=dist_fn,
                strictly_hard_negative=random.choices(
                    population=[True, False], weights=[weight_negative, weight_positive]
                )[0],
            )

            if chosen_index is not None:
                # calculate the distance between the anchor and the chosen negative wihtin the computation graph
                distance_on_choosen_negative = dist_fn(
                    embeddings[a_idx].unsqueeze(dim=0),
                    embeddings[tn_indexes[chosen_index]].unsqueeze(dim=0),
                )
                retained_negative_distances.append(distance_on_choosen_negative)
                # recall the positive distance between the anchor and the positive sample withing the computation graph
                distance_on_positive = dist_fn(
                    embeddings[a_idx].unsqueeze(dim=0),
                    p_embeddings[a_idx].unsqueeze(dim=0),
                )
                retained_positive_distances.append(distance_on_positive)
                # Anchor negetive positive indexes
                triplet_indexes.append((a_idx, tn_indexes[chosen_index], a_idx))

        # Returnt the result
        if len(retained_negative_distances) == 0:
            # no anchor has been retained the loss is 0
            return 0, 0, 0
        else:
            # stack the distances to calculate the loss
            retained_negative_distances = torch.stack(
                tensors=retained_negative_distances, dim=0
            )
            retained_positive_distances = torch.stack(
                tensors=retained_positive_distances, dim=0
            )
            loss, pos, neg = loss_fn(
                retained_positive_distances,
                retained_negative_distances,
                margin,
            )
            r_tup = (loss, pos, neg)
            if return_triplet_indexes:
                r_tup = r_tup + (triplet_indexes,)
            return r_tup
