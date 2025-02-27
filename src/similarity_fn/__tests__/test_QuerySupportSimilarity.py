import pytest
import torch

from similarity_fn.QuerySupportSimilarity import QuerySupportSimilarity


@pytest.fixture
def embeddings():
    support_embedding = torch.randn(5, 10)
    query_embedding = torch.randn(10, 10)
    return support_embedding, query_embedding


def test_p_norm_similarity(embeddings):
    support_embedding, query_embedding = embeddings
    sim_fn = QuerySupportSimilarity(similarity_metric="p_norm", p=2.0)
    sim_matrix = sim_fn(support_embedding, query_embedding)
    assert sim_matrix.shape == (10, 5)


def test_cosine_similarity(embeddings):
    support_embedding, query_embedding = embeddings
    sim_fn = QuerySupportSimilarity(similarity_metric="cosine")
    sim_matrix = sim_fn(support_embedding, query_embedding)
    assert sim_matrix.shape == (10, 5)


def test_invalid_similarity_metric():
    with pytest.raises(ValueError):
        QuerySupportSimilarity(similarity_metric="invalid_metric")


def test_forward_method(embeddings):
    support_embedding, query_embedding = embeddings
    sim_fn = QuerySupportSimilarity(similarity_metric="p_norm", p=2.0)
    sim_matrix = sim_fn.forward(support_embedding, query_embedding)
    assert sim_matrix.shape == (10, 5)
