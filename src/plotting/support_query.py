import os

import matplotlib.pyplot as plt


def plot_query_support(
    result_path, support_set, query_set, meta_data, epoch, k_shot, n_way, q_queries
):

    print("Plotting query and support set")

    # If the image is not 2D then it will take the 80th slice of the 3D image
    if support_set.dim() == 5:
        support_img = [img[0].cpu().numpy() for img in support_set[0]]
        query_img = [img[0].cpu().numpy() for img in query_set[0]]
    else:
        support_img = [img[0][50, :, :].cpu().numpy() for img in support_set[0]]
        query_img = [img[0][50, :, :].cpu().numpy() for img in query_set[0]]

    support_labels = meta_data["support_labels"]
    query_labels = meta_data["query_labels"]

    # Determine grid size based n_way
    n_cols = n_way

    # Determine number of rows based on k_shot and q_queries
    n_rows = q_queries + k_shot

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 8))

    # Clear all axes first
    for ax_row in axs:
        for ax in ax_row:
            ax.clear()
            ax.axis("off")

    # Plot support images
    support_row_counter = 0
    support_col_counter = 0

    for label, s_img in zip(support_labels, support_img):

        # Plot the image
        axs[support_row_counter, support_col_counter].imshow(s_img, cmap="gray")
        axs[support_row_counter, support_col_counter].set_title(f"Support: {label}")
        axs[support_row_counter, support_col_counter].axis("on")

        # Add red border around support images
        axs[support_row_counter, support_col_counter].spines["top"].set_color("red")
        axs[support_row_counter, support_col_counter].spines["bottom"].set_color("red")
        axs[support_row_counter, support_col_counter].spines["left"].set_color("red")
        axs[support_row_counter, support_col_counter].spines["right"].set_color("red")
        axs[support_row_counter, support_col_counter].spines["top"].set_linewidth(2)
        axs[support_row_counter, support_col_counter].spines["bottom"].set_linewidth(2)
        axs[support_row_counter, support_col_counter].spines["left"].set_linewidth(2)
        axs[support_row_counter, support_col_counter].spines["right"].set_linewidth(2)

        # Update counters
        support_row_counter += 1
        if support_row_counter == k_shot:
            support_row_counter = 0
            support_col_counter += 1

    # Plot query images
    query_row_start = k_shot  # Start after support images
    query_col_counter = 0
    current_row = query_row_start

    for label, q_img in zip(query_labels, query_img):

        # Plot the image
        axs[current_row, query_col_counter].imshow(q_img, cmap="gray")
        axs[current_row, query_col_counter].set_title(f"Query: {label}")
        axs[current_row, query_col_counter].axis("on")

        # Add green border around query images
        axs[current_row, query_col_counter].spines["top"].set_color("green")
        axs[current_row, query_col_counter].spines["bottom"].set_color("green")
        axs[current_row, query_col_counter].spines["left"].set_color("green")
        axs[current_row, query_col_counter].spines["right"].set_color("green")
        axs[current_row, query_col_counter].spines["top"].set_linewidth(2)
        axs[current_row, query_col_counter].spines["bottom"].set_linewidth(2)
        axs[current_row, query_col_counter].spines["left"].set_linewidth(2)
        axs[current_row, query_col_counter].spines["right"].set_linewidth(2)

        # Update counters
        query_col_counter += 1
        if query_col_counter == n_way:
            query_col_counter = 0
            current_row += 1

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, f"query_support_epoch_{epoch}.png"))
    plt.close(fig)
