import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.widgets import Button
from sklearn.manifold import TSNE

from plotting.utils import rerange_image


def plot_tsne(
    result_path,
    anchor_embeddings,
    positive_embeddings,
    epoch,
    links=None,
    anchor_imgs=None,
    positive_imgs=None,
    dims=2,
    group_by=None,
    anchor_meta_data=False,
    positive_meta_data=False,
):
    if group_by is not None and not anchor_meta_data:
        raise ValueError("Anchor meta data is required for group_by")
    if group_by is not None and not positive_meta_data:
        raise ValueError("Positive meta data is required for group_by")

    fig = plt.figure(figsize=(10, 8))
    if dims in [3, 2]:
        ax, a_embed, p_embed = plot_nd_tsne(
            fig=fig,
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            links=links,
            dims=dims,
            group_by=group_by,
            anchor_meta_data=anchor_meta_data,
            positive_meta_data=positive_meta_data,
        )
    else:
        raise ValueError("Invalid dimensions for TSNE plot")

    if anchor_imgs is not None or positive_imgs is not None:
        if dims != 2:
            raise ValueError("Images can only be plotted in 2D")
        else:
            add_images(
                ax=ax,
                a_embed=a_embed,
                p_embed=p_embed,
                anchor_imgs=anchor_imgs,
                positive_imgs=positive_imgs,
            )

    filename_dims = str(dims) + "d"
    filename_end = "" if epoch is None else f"_epoch_{epoch}"
    # Save the plot
    plt.savefig(f"{result_path}/tsne_{filename_dims}_{filename_end}.pdf")
    if dims == 2:
        pickle.dump(
            fig,
            open(f"{result_path}/tsne_{filename_dims}_{filename_end}.pickle", "wb"),
        )
    plt.close()


def plot_nd_tsne(
    fig,
    anchor_embeddings,
    positive_embeddings,
    links=None,
    dims=3,
    group_by=None,
    anchor_meta_data=None,
    positive_meta_data=None,
):
    if anchor_embeddings[0].shape[0] > 3:
        print("Plotting with TSNE")
        data = np.concatenate((anchor_embeddings, positive_embeddings), axis=0)

        # Perplexity works well at 30. For super small samples < 30, we just do n_samples/3 - 1
        n_samples = data.shape[0]
        perplexity = min(30, n_samples / 3 - 1)

        tsne = TSNE(
            n_components=dims,
            perplexity=perplexity,
            max_iter=500,
            random_state=0,
            init="pca",
            method="exact",  # Changed this to exact, because it is more stable than barnes
        )
        embedding = tsne.fit_transform(data)

        # Split the embedding for the two sets of points
        a_embed = embedding[: anchor_embeddings.shape[0]]
        p_embed = embedding[anchor_embeddings.shape[0] :]

    else:
        print("Plotting without TSNE")
        a_embed = anchor_embeddings
        p_embed = positive_embeddings

    # Create 3D plot
    if dims == 3:
        ax = fig.add_subplot(111, projection="3d")
    elif dims == 2:
        ax = fig.add_subplot(111)

    # Get the x, y, z coordinates
    a_xyz = [a_embed[:, i] for i in range(dims)]
    p_xyz = [p_embed[:, i] for i in range(dims)]

    if group_by is None:
        # ---- Simple tsne plot ----
        # Plot first list in green
        ax.scatter(*a_xyz, c="green", label="Anchors")
        # Plot second list in blue
        ax.scatter(*p_xyz, c="blue", label="Positives")
    else:
        # ---- Grouped tsne plot ----
        # Get wanted attribute to group by for support and query set
        a_group = [
            anchor_meta_data[idx][group_by["modality"]][group_by["attribute"]]
            for idx in range(len(anchor_meta_data))
        ]
        p_group = [
            positive_meta_data[idx][group_by["modality"]][group_by["attribute"]]
            for idx in range(len(positive_meta_data))
        ]

        # Get unique values of the attribute
        unique_values = list(set(a_group + p_group))
        # turn all values into item is they are tensors
        unique_values = [
            item.item() if hasattr(item, "item") else item for item in unique_values
        ]
        unique_values.sort()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_values)))
        color_map = {unique_values[i]: colors[i] for i in range(len(unique_values))}

        # Get colors for the support set and query set
        a_colors = [
            color_map[a_group[i].item() if hasattr(a_group[i], "item") else a_group[i]]
            for i in range(len(a_group))
        ]
        p_colors = [
            color_map[p_group[i].item() if hasattr(p_group[i], "item") else p_group[i]]
            for i in range(len(p_group))
        ]

        # Scatter plot for the support set and query set
        ax.scatter(*a_xyz, c=a_colors, marker="x", label="Anchors")
        ax.scatter(*p_xyz, c=p_colors, marker="o", label="Positives")

        # Legend for groupings

        grouping_legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                label=grouping,
                markersize=8,
            )
            for grouping, color in color_map.items()
        ]

        second_legend = plt.legend(
            handles=grouping_legend_elements,
            title=group_by["attribute"],
            loc="upper left",
            bbox_to_anchor=(-0.2, 1.15),
        )

        ax.add_artist(second_legend)

    if links is None:
        # anchor and positive embeddings are connected by order in the list
        for i in range(len(a_embed)):
            line_xyz = [[a_embed[i, j], p_embed[i, j]] for j in range(dims)]
            # draw connecting lines
            ax.plot(
                *line_xyz,
                c="red",
                alpha=0.3,
            )
    elif type(links) is dict:
        # iterate through the links and draw the lines
        for a_index, p_indexes in links.items():
            for p_index in p_indexes:
                line_xyz = [
                    [a_embed[a_index, j], p_embed[p_index, j]] for j in range(dims)
                ]
                ax.plot(
                    *line_xyz,
                    c="lime",
                    alpha=0.3,
                )
    else:
        # iterate through the links and draw the lines
        for i, link in enumerate(links):
            if link == i:
                # retrieved the correct link
                color = "lime"
            else:
                # retrieved the wrong link
                color = "black"
            line_xyz = [[a_embed[link, j], p_embed[i, j]] for j in range(dims)]
            ax.plot(
                line_xyz,
                c=color,
                alpha=0.3,
            )

    # Add title and legend
    if dims == 2:
        ax.set_title("2D TSNE Projection")
    else:
        ax.set_title("3D TSNE Projection")
    ax.legend()

    # Add labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if dims == 3:
        ax.set_zlabel("Z")

    return ax, a_embed, p_embed


def add_images(ax, a_embed, p_embed, anchor_imgs=None, positive_imgs=None):
    """
    Plots the embeddings in a 3D TSNE space. Giving the anchor and corresponding positive embeddings the same id in the
    labels list will make them appear as connected points in the plot.

    Args:
        ax: Axis object
        a_embed: Embeddings of the anchor images
        p_embed: Embeddings of the positive images
        anchor_imgs: List of anchor images
        positive_images: List of positive images
    """
    # List to store the image artists
    image_artists = []

    img_embed_pairs = []
    if anchor_imgs is not None:
        img_embed_pairs.append([zip(anchor_imgs, a_embed), dict(edgecolor="green")])
    if positive_imgs is not None:
        img_embed_pairs.append([zip(positive_imgs, p_embed), dict(edgecolor="blue")])

    # I didn't get what this is for
    def add_images(img_embed_pairs):
        for img_emb, color in img_embed_pairs:
            for img, embed in img_emb:
                if len(img.shape) == 4:
                    img = img.squeeze(0)
                    img = img[50, :, :]
                img = rerange_image(img)
                img = np.moveaxis(img, 0, -1)
                offset_img = OffsetImage(img, zoom=0.25, cmap="gray")
                ab = AnnotationBbox(
                    offset_img, (embed[0], embed[1]), frameon=True, bboxprops=color
                )
                image_artists.append(ab)
                ax.add_artist(ab)

    # Initially add images to the scatter plot
    add_images(img_embed_pairs)

    # Define toggle function
    def toggle_images(event):
        visibility = not image_artists[0].get_visible()
        for artist in image_artists:
            artist.set_visible(visibility)
        plt.draw()

    # Adding a button
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
    button = Button(ax_button, "Toggle Images")
    button.on_clicked(toggle_images)
