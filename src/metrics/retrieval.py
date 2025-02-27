import numpy as np


def mean_recall_at_k(k: int, ordered_rel: np.ndarray):
    """
    Calculate the recall at k for a given query.

    Args:
        k (int): k value for selecting the top k supports
        ordered_rel (np.ndarray): shape (n_querries, n_supports)
            np.ndarray of binary values indicating the relevancy of each support
            0 for non-relevant and 1 for relevant
            ordered by decreasing similarity

    Returns:
        float: mean recall at k over all queries

    Example:
        >>> rel = np.array([[1, 0, 1, 0],
                            [0, 0, 1, 1]])
        >>> mean_recall_at_k(2, rel)
        0.25
    """
    # Perform some checks
    assert np.all(
        np.sum(ordered_rel, axis=1) == k
    ), "k must be equal to the number of relevant supports"
    # Calculate the recall at k for each query
    return np.mean(np.sum(ordered_rel[:, :k], axis=1) / k)


def mean_hit_at_k(k: int, ordered_rel: np.ndarray):
    """
    Calculates the boolean hit value for each query looking at the top k supports.
    The hit value is 1 if at least one relevant support is in the top k supports.
    The mean is taken over all queries.

    Args:
        k (int): k value for selecting the top k supports
        ordered_rel (np.ndarray): shape (n_querries, n_supports)
            np.ndarray of binary values indicating the relevancy of each support
            0 for non-relevant and 1 for relevant
            ordered by decreasing similarity

    Returns:
        float: mean hit rate at k over all queries

    Example:
        >>> rel = np.array([[1, 0, 1, 0],
                            [0, 0, 1, 1]])
        >>> mean_hit_at_k(2, rel)
        0.5
    """
    # Calculate the hit rate at k for each query
    return np.mean(np.any(ordered_rel[:, :k], axis=1))


def mean_majority_vote_hit_at_k(
    k: int,
    relevant_labels: np.array,
    labels: np.ndarray,
):
    """
    Calcultes a majority vote on the retrieved top k labels ordered by similarity for each query.
    The majority vote is calculated by taking the most frequent label in the top k labels.
    If the majority vote is equal to the relevant label, the hit value is 1, otherwise 0.
    The mean is taken over all queries.

    Args:
        k (int): k value for selecting the top k supports
        relavant_labels (np.array): the relevant label for each query
        labels (np.ndarray): shape (n_querries, n_supports)
            np.ndarray of labels for each support
            ordered by decreasing similarity

    Returns:
        float: mean hit rate at k over all queries

    Example:
        >>> labels = np.array([[0, 0, 2, 0, 1, 2],
                               [1, 0, 2, 2, 1, 0],
                               [2, 1, 2, 0, 0, 1],
                               [1, 1, 2, 0, 0, 2]])
        >>> relavant_label = np.array([0, 0, 2, 1])
        >>> mean_majority_vote_hit_at_k(2, relavant_label, labels)
        0.75 # hit, miss, hit, hit
    """
    # Perform some checks
    assert (
        relevant_labels.shape[0] == labels.shape[0]
    ), "relevant_labels and ordered_labels must have the same length"
    n_queries = labels.shape[0]

    # Get the top k labels for each query
    top_k_labels = labels[:, :k]

    # Get the unique labels found in the batch
    # and create a mapping from unique label index to the label
    unique_labels = np.unique(top_k_labels)
    unique_index_to_label = {i: lab for i, lab in enumerate(unique_labels)}

    # Count the votes for each query
    # votes_per_query is a matrix of shape (n_queries, n_unique_labels)
    votes_per_query = np.zeros((n_queries, len(unique_labels)))
    for i, lab in enumerate(unique_labels):
        # count the number of votes for each label with respect to each query
        votes_per_query[:, i] = np.sum(top_k_labels == lab, axis=1)

    # Get the max votes value for each query
    max_votes_per_query = np.max(votes_per_query, axis=1)

    # Get the indexes of the labels with the max votes for each query
    indexes_votes_per_query = [
        np.where(votes_per_query[i] == max_votes_per_query[i])[0]
        for i in range(len(max_votes_per_query))
    ]

    # Go through each query and get the majority vote label
    selected = np.zeros(n_queries)
    for i in range(n_queries):
        if len(indexes_votes_per_query[i]) > 1:
            # Multiple labels have the same frequency
            # therefore taking the one with the highest similarity,
            # i.e. the lowest index available for the label

            # Create an array to store the indexes of the labels with the max
            # votes in the ranked labels
            indexes = np.zeros(len(indexes_votes_per_query[i]))

            for idx, lab in enumerate(indexes_votes_per_query[i]):
                # for each label with max votes find the first index in the ranked labels
                # the lowest index is the one that will be taken as it is the most similar
                # to the query image. It could be that there is a label with even higher
                # similarity but it is not taken because it got excluded by the majority vote
                indexes[idx] = np.min(
                    np.where(top_k_labels[i] == unique_index_to_label[lab])
                )

            # Get the selected label with the lowest index
            selected_label = unique_index_to_label[
                indexes_votes_per_query[i][np.argmin(indexes)]
            ]

        else:
            selected_label = unique_index_to_label[indexes_votes_per_query[i][0]]

        # Store the selected label for the query
        selected[i] = selected_label

    # Calculate the hit rate
    return np.mean(selected == relevant_labels)


def get_ranked_retrieval_matrix(support_labels: np.array, similarities: np.array):
    """
    Get the ranked retrieval matrix for each query.

    Args:
        support_labels (np.array): array of labels for each support of shape (n_supports)
        similarities (np.array): shape (n_querries, n_supports)
            np.ndarray of similarity values for each support with respect to the each query

    Returns:
        np.array: ranked retrieval matrix of shape (n_querries, n_supports)

    Example:
        >>> labels = np.array([0, 1, 2, 3, 4, 1])
        >>> similarities = np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                     [0.8, 0.9, 0.7, 0.6, 0.5, 0.4],
                                     [0.7, 0.8, 0.9, 0.6, 0.5, 0.4],
                                     [0.6, 0.7, 0.8, 0.9, 0.5, 0.4]])
        >>> get_ranked_retrieval_matrix(labels, similarities)
        array([[0, 1, 2, 3, 4, 1],
               [1, 0, 2, 3, 4, 1],
               [2, 1, 0, 3, 4, 1],
               [3, 2, 1, 0, 4, 1]])
    """
    # Duplicate the labels for each query
    n_queries = similarities.shape[0]
    query_labels = np.tile(support_labels, (n_queries, 1))

    # Order the labels by the similarities
    ordered_labels = order_by_similarities(query_labels, similarities)

    return ordered_labels


def get_relevancy_matrix(query_labels: np.array, support_labels: np.array):
    """
    Get the relevancy matrix for each support to each query.

    Args:
        query_labels (np.array): array of relevant labels for each query
            of shape (n_queries)
        support_labels (np.array): array of labels for each support of shape (n_supports)

    Returns:
        np.array: relevancy matrix of shape (n_queries, n_supports)

    Example:
        >>> labels = np.array([0, 0, 2, 0, 1, 2])
        >>> query_labels = np.array([0, 1, 2, 1])
        >>> get_relevancy_matrix(query_labels, labels)
        array([[1, 1, 0, 1, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 1],
               [0, 0, 0, 0, 1, 0]])
    """
    # Create a relevancy matrix
    n_queries = query_labels.shape[0]
    n_supports = support_labels.shape[0]
    relevancy = np.zeros((n_queries, n_supports), dtype=bool)

    for i, rel in enumerate(query_labels):
        # Get the relevancy of each support with respect to the query
        relevancy[i, :] = support_labels == rel

    return relevancy.astype(int)


def order_by_similarities(array: np.ndarray, similarities: np.ndarray):
    """
    Order the matrix by the similarity matrix.

    Args:
        array (np.ndarray): shape (n_querries, n_supports)
            array to be sorted by the similarities
        similarities (np.ndarray): shape (n_querries, n_supports)
            np.ndarray of similarity values for each support with respect to the each query

    Returns:
        (np.ndarray): ordered matrix with shape (n_querries, n_supports)

    Example:
        >>> array = np.array([[1, 0, 1], [0, 1, 1]])
        >>> similarities = np.array([[0.9, 0.8, 0.7], [0.6, 0.8, 0.7]])
        >>> order_array_by_rank(array, similarities)
        array([[1, 0, 1],
               [1, 1, 0]])
    """
    # Perform some checks
    assert len(array) == len(
        similarities
    ), "array matrix and similarities must have the same length"

    # Order the relevancy matrix by the similarity matrix in descending order
    order = np.argsort(similarities, axis=1)

    # Return the ordered relevancy matrix
    return np.take_along_axis(array, order, axis=1)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
