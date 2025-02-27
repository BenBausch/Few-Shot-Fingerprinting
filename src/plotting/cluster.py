import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def plot_cluster_tsne(
    embeddings, labels, result_path, dims=2, perplexity=20, animated=False
):
    assert (not animated) or (animated and dims == 3), "Animation only supported for 3D"
    # Fit the TSNE model
    tsne = TSNE(
        n_components=dims,
        perplexity=perplexity,
        max_iter=500,
        random_state=0,
        method="barnes_hut",
    )
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Initialize the figure
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(10, 8))
    if dims == 3:
        ax = fig.add_subplot(111, projection="3d")
    elif dims == 2:
        ax = fig.add_subplot(111)

    min_samples_per_subject = 100

    # Plot the embeddings grouped by the labels with a color unique to each label
    # Transform the labels to integers for color mapping and indexing
    label_set = sorted(list(set(labels)))
    labels_to_int = {label: i for i, label in enumerate(label_set)}
    labels_int = np.array([labels_to_int[label] for label in labels])
    gray_colors = sns.light_palette("#ebd4b2", n_colors=len(label_set), as_cmap=True)
    colors = sns.color_palette("rocket", as_cmap=True)

    # First plot the background points with low alpha
    for label in label_set:
        l_int = labels_to_int[label]
        i_embed = tsne_embeddings[labels_int == l_int]
        i_xyz = [i_embed[:, i] for i in range(dims)]

        if i_embed.shape[0] <= min_samples_per_subject:
            color = gray_colors(labels_to_int[label] / len(label_set))
            border = None
            alpha = 0.1
            scale = 25
            rand_marker = np.random.choice(["o", "s", "x", "D", "P"])
            data = pd.DataFrame(i_xyz).T
            # random color for the border
            if dims == 3:
                # saeborn can not do 3d scatter plot
                ax.scatter(
                    data[0],
                    data[1],
                    data[2],
                    c=[color],
                    edgecolors=border,
                    alpha=alpha,
                    s=scale,
                    marker=rand_marker,
                )
            else:
                sns.scatterplot(
                    data=data,
                    x=0,
                    y=1,
                    color=color,
                    edgecolor=border,
                    alpha=alpha,
                    s=scale,
                    marker=rand_marker,
                )

    # Then plot the foreground points with high alpha
    for label in label_set:
        l_int = labels_to_int[label]
        i_embed = tsne_embeddings[labels_int == l_int]
        i_xyz = [i_embed[:, i] for i in range(dims)]

        if i_embed.shape[0] > min_samples_per_subject:
            print(f"Plotting {label} with {i_embed.shape[0]} samples")
            print(l_int)
            color = colors(labels_to_int[label] / len(label_set))
            # boder color should be from the same color map, but a bit shifted
            shift = int(len(label_set) // 10)
            if l_int < len(label_set) - shift:
                border = colors((l_int + shift) / len(label_set))
            else:
                border = colors((l_int + shift - len(label_set)) / len(label_set))
            alpha = 1
            scale = 80
            rand_marker = np.random.RandomState(l_int).choice(["o", "s", "x", "D", "P"])
            data = pd.DataFrame(i_xyz).T
            # random color for the border
            if dims == 3:
                # saeborn can not do 3d scatter plot
                ax.scatter(
                    data[0],
                    data[1],
                    data[2],
                    c=[color],
                    edgecolors=border,
                    alpha=alpha,
                    s=scale,
                    marker=rand_marker,
                )
            else:
                sns.scatterplot(
                    data=data,
                    x=0,
                    y=1,
                    color=color,
                    edgecolor=border,
                    alpha=alpha,
                    s=scale,
                    marker=rand_marker,
                )

    if animated:
        result_path = os.path.join(os.path.dirname(result_path), "tsne_animation")
        os.makedirs(result_path, exist_ok=True)
        # Rotate the 3D plot
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(0.001)
            # Save the plot as a gif
            plt.savefig(os.path.join(result_path, f"{angle}.svg", format="svg"))
    else:
        # Save the plot
        plt.xticks([])
        plt.yticks([])
        if dims == 3:
            ax.set_zticks([])
        plt.ylabel("Dimention 2")
        plt.xlabel("Dimention 1")
        plt.savefig(result_path, format="svg")
        plt.savefig(result_path.replace(".svg", ".png"), format="png")
