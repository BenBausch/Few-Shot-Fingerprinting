from math import sqrt
from unittest.mock import call

import pytest
import torch

from losses.CustomTripletMarginLoss import CustomTripletMarginLoss


def test_get_true_negative_indexes():
    # ---- No Duplicate Embeddings ----
    def test_no_duplicate_embeddings():
        embedding_ids = ["a", "b", "c", "d", "e"]
        result = CustomTripletMarginLoss._get_true_negative_indexes(embedding_ids)
        assert result == [
            [1, 2, 3, 4],
            [0, 2, 3, 4],
            [0, 1, 3, 4],
            [0, 1, 2, 4],
            [0, 1, 2, 3],
        ]

    def test_duplicate_embeddings():
        embedding_ids = ["a", "b", "a", "b", "a"]
        result = CustomTripletMarginLoss._get_true_negative_indexes(embedding_ids)
        assert result == [[1, 3], [0, 2, 4], [1, 3], [0, 2, 4], [1, 3]]

    def test_no_true_negatives():
        embedding_ids = ["a", "a", "a", "a", "a"]
        result = CustomTripletMarginLoss._get_true_negative_indexes(embedding_ids)
        assert result == [[], [], [], [], []]

    def test_single_embedding():
        embedding_ids = ["a"]
        result = CustomTripletMarginLoss._get_true_negative_indexes(embedding_ids)
        assert result == [[]]

    def test_empty_embedding_list():
        embedding_ids = []
        result = CustomTripletMarginLoss._get_true_negative_indexes(embedding_ids)
        assert result == []

    test_no_duplicate_embeddings()
    test_duplicate_embeddings()
    test_no_true_negatives()
    test_single_embedding()
    test_empty_embedding_list()


def test_loss_fn():
    def test_positive_equal_negative():
        positive_distances = torch.tensor(data=[1, 2, 3, 4], dtype=torch.float32)
        negative_distances = torch.tensor(data=[1, 2, 3, 4], dtype=torch.float32)
        margin = 1
        result = CustomTripletMarginLoss.loss_fn(
            positive_distances=positive_distances,
            negative_distances=negative_distances,
            margin=margin,
        )
        assert result == 1.0

    def test_positive_greater_than_negative():
        positive_distances = torch.tensor(data=[2, 3, 4, 5], dtype=torch.float32)
        negative_distances = torch.tensor(data=[1, 2, 3, 4], dtype=torch.float32)
        margin = 1
        result = CustomTripletMarginLoss.loss_fn(
            positive_distances=positive_distances,
            negative_distances=negative_distances,
            margin=margin,
        )
        assert result == 2

    test_positive_equal_negative()
    test_positive_greater_than_negative()


def test_mine_semi_hard_negatives(mocker):

    def test_no_semi_hard_negatives():
        loss = CustomTripletMarginLoss(margin=1.5, p_norm=2)
        anchor_embedding = torch.tensor(
            data=[1, 1], dtype=torch.float32, requires_grad=True
        )
        negative_embeddings = torch.tensor(
            data=[[6, 6], [7, 7], [-5, -5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        positive_distance = torch.tensor(data=0.5, dtype=torch.float32)
        np_mock = mocker.patch("numpy.random.choice")
        result = loss.mine_semi_hard_negative(
            a_embedding=anchor_embedding,
            negative_embeddings=negative_embeddings,
            positive_distance=positive_distance,
            dist_fn=loss.dist_fn,
            margin=loss.margin,
        )
        np_mock.assert_not_called()
        assert result is None

    def test_single_semi_hard_negative(mocker):

        loss = CustomTripletMarginLoss(margin=1.5, p_norm=2)
        anchor_embedding = torch.tensor(
            data=[1, 1], dtype=torch.float32, requires_grad=True
        )
        negative_embeddings = torch.tensor(
            data=[[6, 6], [2, 2], [7, 7], [-5, -5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        positive_distance = torch.tensor(data=0.5, dtype=torch.float32)

        np_mock = mocker.patch("numpy.random.choice", return_value=1)

        result = loss.mine_semi_hard_negative(
            a_embedding=anchor_embedding,
            negative_embeddings=negative_embeddings,
            positive_distance=positive_distance,
            dist_fn=loss.dist_fn,
            margin=loss.margin,
        )
        np_mock.assert_called_once_with([1])
        assert result == 1

    def test_multiple_semi_hard_negatives(mocker):

        loss = CustomTripletMarginLoss(margin=2, p_norm=2)
        anchor_embedding = torch.tensor(
            data=[1, 1], dtype=torch.float32, requires_grad=True
        )
        negative_embeddings = torch.tensor(
            data=[[1.5, 1.5], [2, 2], [7, 7], [-0.5, -0.5]],
            dtype=torch.float32,
            requires_grad=True,
        )
        positive_distance = torch.tensor(data=0.5, dtype=torch.float32)

        np_mock = mocker.patch("numpy.random.choice", return_value=1)

        result = loss.mine_semi_hard_negative(
            a_embedding=anchor_embedding,
            negative_embeddings=negative_embeddings,
            positive_distance=positive_distance,
            dist_fn=loss.dist_fn,
            margin=loss.margin,
        )
        np_mock.assert_called_once_with([0, 1, 3])
        assert result == 1

    test_no_semi_hard_negatives()
    test_single_semi_hard_negative(mocker)
    test_multiple_semi_hard_negatives(mocker)


def test_semi_hard_loss(mocker):

    def test_same_subjects():
        loss = CustomTripletMarginLoss(margin=1.5, p_norm=2)
        embeddings = torch.tensor(
            data=[[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float32
        )
        p_embeddings = torch.tensor(
            data=[[1.2, 1.2], [2.3, 2.3], [3.4, 3.4], [4.5, 4.5]], dtype=torch.float32
        )
        embedding_ids = ["1", "1", "1", "1"]
        result = loss.semi_hard_loss(
            dist_fn=loss.dist_fn,
            margin=loss.margin,
            loss_fn=loss.loss_fn,
            embeddings=embeddings,
            p_embeddings=p_embeddings,
            embedding_ids=embedding_ids,
        )
        # As no negative samples are available, the loss should be 0
        assert result == (0, 0, 0)

    def test_only_unique_subjects_no_semi_hard():
        loss = CustomTripletMarginLoss(margin=0, p_norm=2)
        embeddings = torch.tensor(
            data=[[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float32
        )
        p_embeddings = torch.tensor(
            data=[[1.2, 1.2], [2.3, 2.3], [3.4, 3.4], [4.5, 4.5]], dtype=torch.float32
        )
        embedding_ids = ["1", "2", "3", "4"]
        result = loss.semi_hard_loss(
            dist_fn=loss.dist_fn,
            margin=loss.margin,
            loss_fn=loss.loss_fn,
            embeddings=embeddings,
            p_embeddings=p_embeddings,
            embedding_ids=embedding_ids,
        )
        assert result == (0, 0, 0)

    def test_only_unique_subjects_single_semi_hard():
        loss = CustomTripletMarginLoss(margin=1.5, p_norm=2)
        embeddings = torch.tensor(
            data=[[1, 1], [2, 2], [6, 6], [8, 8]], dtype=torch.float32
        )
        p_embeddings = torch.tensor(  # perfect positive embeddings
            data=[[1, 1], [2, 2], [6, 6], [8, 8]], dtype=torch.float32
        )
        embedding_ids = ["1", "2", "3", "4"]
        result = loss.semi_hard_loss(
            dist_fn=loss.dist_fn,
            margin=loss.margin,
            loss_fn=loss.loss_fn,
            embeddings=embeddings,
            p_embeddings=p_embeddings,
            embedding_ids=embedding_ids,
        )
        assert torch.allclose(input=result[0], other=torch.tensor(0.0857864), atol=1e-5)

    def test_non_unique_subjects_multiple_semi_hard():
        loss = CustomTripletMarginLoss(margin=1.5, p_norm=2)
        # pairs: (a_1, b) = 1.21716, (b, a_1)=1.21716, (c, a_2)=0.79289, (a_2, c)=0.79289
        embeddings = torch.tensor(
            data=[[1, 1], [1.2, 1.2], [3, 3], [3.5, 3.5]],
            dtype=torch.float32,
        )
        p_embeddings = torch.tensor(  # perfect positive embeddings
            data=[[1, 1], [1.2, 1.2], [3, 3], [3.5, 3.5]],
            dtype=torch.float32,
        )
        embedding_ids = ["a", "b", "a", "c"]
        result = loss.semi_hard_loss(
            dist_fn=loss.dist_fn,
            margin=loss.margin,
            loss_fn=loss.loss_fn,
            embeddings=embeddings,
            p_embeddings=p_embeddings,
            embedding_ids=embedding_ids,
        )
        assert torch.allclose(input=result[0], other=torch.tensor(1.005025), atol=1e-6)
        assert torch.allclose(input=result[1], other=torch.tensor(loss.eps), atol=1e-6)
        assert torch.allclose(
            input=result[2],
            other=torch.tensor((2 * sqrt(0.08) + 2 * sqrt(0.5))) / 4,
            atol=1e-6,
        )

    def test_standard_case(mocker):
        loss = CustomTripletMarginLoss(margin=1.5, p_norm=2)
        embeddings = torch.tensor(
            data=[[1, 1], [2, 2], [3, 3], [4, 4]], dtype=torch.float32
        )
        p_embeddings = torch.tensor(
            data=[[1.2, 1.2], [2.2, 2.2], [3.2, 3.2], [4.2, 4.2]], dtype=torch.float32
        )
        embedding_ids = ["1", "2", "3", "4"]
        chosen_index_mock = mocker.patch(
            "losses.CustomTripletMarginLoss.CustomTripletMarginLoss.mine_semi_hard_negative",
            return_value=0,
        )
        chosen_index_calls = [
            call(
                a_embedding=embeddings[0],
                negative_embeddings=embeddings[1:],
                positive_distance=torch.tensor(sqrt(0.08)),
                dist_fn=loss.dist_fn,
                margin=loss.margin,
            ),
            call(
                a_embedding=embeddings[1],
                negative_embeddings=torch.cat((embeddings[0:1], embeddings[2:])),
                positive_distance=torch.tensor(sqrt(0.08)),
                dist_fn=loss.dist_fn,
                margin=loss.margin,
            ),
            call(
                a_embedding=embeddings[2],
                negative_embeddings=torch.cat((embeddings[0:2], embeddings[3:])),
                positive_distance=torch.tensor(sqrt(0.08)),
                dist_fn=loss.dist_fn,
                margin=loss.margin,
            ),
            call(
                a_embedding=embeddings[3],
                negative_embeddings=embeddings[:3],
                positive_distance=torch.tensor(sqrt(0.08)),
                dist_fn=loss.dist_fn,
                margin=loss.margin,
            ),
        ]
        result = loss.semi_hard_loss(
            dist_fn=loss.dist_fn,
            margin=loss.margin,
            loss_fn=loss.loss_fn,
            embeddings=embeddings,
            p_embeddings=p_embeddings,
            embedding_ids=embedding_ids,
        )
        assert chosen_index_mock.call_count == 4
        for recorded_call, expected_call in zip(
            chosen_index_mock.call_args_list, chosen_index_calls
        ):
            for recorded_arg, expected_arg in zip(
                recorded_call.kwargs, expected_call.kwargs
            ):
                if isinstance(recorded_call.kwargs[recorded_arg], torch.Tensor):
                    assert torch.allclose(
                        input=recorded_call.kwargs[recorded_arg],
                        other=expected_call.kwargs[expected_arg],
                        atol=1e-6,
                    )
        # only the distances between "1" and "2" are semi-hard
        assert torch.allclose(input=result[0], other=torch.tensor(0.1843145), atol=1e-9)
        assert torch.allclose(input=result[2], other=torch.tensor(2.47485), atol=1e-5)
        assert torch.allclose(
            input=result[1], other=torch.tensor(sqrt(0.08)), atol=1e-6
        )

    test_same_subjects()
    test_only_unique_subjects_no_semi_hard()
    test_only_unique_subjects_single_semi_hard()
    test_non_unique_subjects_multiple_semi_hard()
    test_standard_case(mocker)


def test_all_losses():
    loss = CustomTripletMarginLoss(margin=2, p_norm=2)
    embeddings = torch.randn(10, 5)
    p_embeddings = torch.randn(10, 5)
    embedding_ids = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    all_losses = loss.all_losses(embeddings, p_embeddings, embedding_ids)
    assert isinstance(all_losses, tuple)
    assert len(all_losses) == 3
    assert isinstance(all_losses[0], torch.Tensor)
    assert isinstance(all_losses[1], torch.Tensor)
    assert isinstance(all_losses[2], torch.Tensor)


if __name__ == "__main__":
    pytest.main()
