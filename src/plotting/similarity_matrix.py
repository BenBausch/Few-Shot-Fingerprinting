from matplotlib import pyplot as plt


def plot_similiarity_matrix(matrix, result_path, highlight_row=True):
    """
    Plot the similarity matrix, with a highlight of the minimal value in each row.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap="viridis")
    fig.colorbar(cax)

    if highlight_row:
        for i in range(matrix.shape[0]):
            # for each minimum cell per row, color the cell borders red
            min_index = matrix[i].argmin()
            ax.add_patch(
                plt.Rectangle(
                    (min_index - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="red",
                    lw=2,
                )
            )

    plt.savefig(result_path)
    plt.close()
