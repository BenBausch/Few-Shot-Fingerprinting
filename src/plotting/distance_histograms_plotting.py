import pickle
import time
import os

import tqdm
from src.inference.Embedder import Embedder
from src.utils.utils import get_dataloaders, setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "CMU Serif"
import seaborn as sns

config_file = "path/to/config.json"

(
    config,
    checkpoint,
    model,
    optimizer,
    scheduler,
    loss_function,
    batch_size,
    number_of_epochs,
    device,
    result_path,
) = setup(config_file)

embed_loader = get_dataloaders(config=config, batch_size=batch_size)
embedder = Embedder(
   config=config,
   model=model,
   data_loader=embed_loader,
   device=device,
   result_path=config["EMBEDDER"]["RESULT_PATH"],
   max_batch_size=200,
)
embedder.embed_data()


# Load the pickle file


def inspect_and_load_pickle(file_path, distance_to_centroids=False):
    with open(file_path, "rb") as file:
        data = pickle.load(file)

    df = pd.DataFrame(data).T
    df = df.reset_index().rename(columns={"index": "subject_id"})

    # Put subject ids and embeddings where len(embeddings list) < 1 in a dictionary. Keep these for later
    one_embedding_subjects = df[df["embeddings"].map(len) == 1][
        ["subject_id", "embeddings"]
    ].to_dict(orient="records")

    # drop rows where len(embeddings list) < 1
    df = df[df["embeddings"].map(len) > 1]
    print(f"Number of rows with more than 1 embedding: {len(df)}")

    # Calculate for each row the mean embedding. Store these values in new columns and drop the embeddings column
    df["mean_embedding"] = df["embeddings"].apply(lambda x: np.mean(x, axis=0))

    subjects_mean_embeddings = df[["subject_id", "mean_embedding"]].to_dict(
        orient="records"
    )

    # Save subjects_mean_embeddings to a file that I can open later as a dictionary\
    with open("subjects_mean_embeddings_test.pkl", "wb") as file:
        pickle.dump(subjects_mean_embeddings, file)

    # Calculate the distance between the mean embedding and each of the embeddings in the list and add these values to a new column
    df["distances"] = df.apply(
        lambda row: [
            np.linalg.norm(vector - row["mean_embedding"])
            for vector in row["embeddings"]
        ],
        axis=1,
    )

    if distance_to_centroids == False:
        # Calculate average distance (optional)
        df["mean_distance"] = df["distances"].apply(np.mean)

        mean_embeddings_dict = dict(zip(df["subject_id"], df["mean_embedding"]))

        mean_distances = {}
        subject_ids = df["subject_id"].tolist()

        for i, subject1 in enumerate(tqdm.tqdm(subject_ids)):
            mean1 = mean_embeddings_dict[subject1]
            for subject2 in subject_ids[i + 1 :]:
                mean2 = mean_embeddings_dict[subject2]
                dist = np.linalg.norm(mean1 - mean2)
                mean_distances[(subject1, subject2)] = dist
                mean_distances[(subject2, subject1)] = dist

        # Calculate mean distance between this subject's mean and all other means
        df["mean_to_other_means"] = df["subject_id"].apply(
            lambda sid: np.mean(
                [mean_distances[(sid, other)] for other in subject_ids if other != sid]
            )
        )

        # print mean mean_distance and mean mean_to_other_means
        print(f"Mean mean_distance: {df['mean_distance'].mean()}")
        print(f"Standard deviation mean_distance: {df['mean_distance'].std()}")
        print(f"Max mean_distance: {df['mean_distance'].max()}")
        print(f"Min mean_distance: {df['mean_distance'].min()}")
        print(f"Median mean_distance: {df['mean_distance'].median()}")
        print(f"Mean mean_to_other_means: {df['mean_to_other_means'].mean()}")
        print(
            f"Standard deviation mean_to_other_means: {df['mean_to_other_means'].std()}"
        )
        print(f"Max mean_to_other_means: {df['mean_to_other_means'].max()}")
        print(f"Min mean_to_other_means: {df['mean_to_other_means'].min()}")
        print(f"Median mean_to_other_means: {df['mean_to_other_means'].median()}")

        # Get the directory and base filename
        directory = os.path.dirname(file_path)
        base_filename = os.path.basename(file_path)
        base_without_ext = os.path.splitext(base_filename)[0]
        model_info = os.path.basename(directory)
        new_filename = f"{base_without_ext}_{model_info}_hist.svg"

        plot_histograms(df, new_filename)

        return df

    elif distance_to_centroids == True:
        # Process subjects with only one embedding
        print(
            f"Number of subjects with only one embedding: {len(one_embedding_subjects)}"
        )

        # For each subject with one embedding, find the minimum distance to any mean embedding
        for subject in tqdm.tqdm(one_embedding_subjects):
            subject_id = subject["subject_id"]
            embedding = subject["embeddings"][0]  # Get the single embedding

            # Calculate distances to all mean embeddings
            distances = []
            for mean_subject in subjects_mean_embeddings:
                if mean_subject["subject_id"] != subject_id:  # Don't compare to self
                    mean_embedding = mean_subject["mean_embedding"]
                    distance = np.linalg.norm(embedding - mean_embedding)
                    distances.append((mean_subject["subject_id"], distance))

            # Find the minimum distance and the corresponding subject
            if distances:
                min_distance_subject, min_distance = min(distances, key=lambda x: x[1])
                max_distance_subject, max_distance = max(distances, key=lambda x: x[1])
                subject["max_distance_to_mean"] = max_distance
                subject["farthest_mean_subject"] = max_distance_subject
                subject["min_distance_to_mean"] = min_distance
                subject["closest_mean_subject"] = min_distance_subject
            else:
                subject["max_distance_to_mean"] = None
                subject["farthest_mean_subject"] = None
                subject["min_distance_to_mean"] = None
                subject["closest_mean_subject"] = None

        # Create a DataFrame for the one-embedding subjects with their distances
        one_embedding_df = pd.DataFrame(one_embedding_subjects)

        # Print summary statistics
        if len(one_embedding_df) > 0:
            print(
                f"Mean min distance to means: {one_embedding_df['min_distance_to_mean'].mean()}"
            )
            print(
                f"Standard deviation min distance to means: {one_embedding_df['min_distance_to_mean'].std()}"
            )
            print(
                f"Max min distance to means: {one_embedding_df['min_distance_to_mean'].max()}"
            )
            print(
                f"Min min distance to means: {one_embedding_df['min_distance_to_mean'].min()}"
            )
            print(
                f"Median min distance to means: {one_embedding_df['min_distance_to_mean'].median()}"
            )
            print(
                f"Mean max distance to means: {one_embedding_df['max_distance_to_mean'].mean()}"
            )
            print(
                f"Standard deviation max distance to means: {one_embedding_df['max_distance_to_mean'].std()}"
            )
            print(
                f"Max max distance to means: {one_embedding_df['max_distance_to_mean'].max()}"
            )
            print(
                f"Min max distance to means: {one_embedding_df['max_distance_to_mean'].min()}"
            )
            print(
                f"Median max distance to means: {one_embedding_df['max_distance_to_mean'].median()}"
            )

            # Optional: Plot histogram of minimum distances
            plt.figure(figsize=(10, 6))
            plt.hist(
                one_embedding_df["min_distance_to_mean"].dropna(), bins=30, alpha=0.7
            )
            plt.title(
                "Histogram of Minimum Distances from Single Embeddings to Mean Embeddings"
            )
            plt.xlabel("Minimum Distance")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            plt.show()

        return one_embedding_df, df


def plot_histograms(df, filename):

    mean_mean_distance = round(df["mean_distance"].mean(), 2)
    standard_deviation_mean_distance = round(df["mean_distance"].std(), 2)
    max_mean_distance = round(df["mean_distance"].max(), 2)
    min_mean_distance = round(df["mean_distance"].min(), 2)
    median_mean_distance = round(df["mean_distance"].median(), 2)
    mean_mean_to_other_means = round(df["mean_to_other_means"].mean(), 2)
    standard_deviation_mean_to_other_means = round(df["mean_to_other_means"].std(), 2)
    max_mean_to_other_means = round(df["mean_to_other_means"].max(), 2)
    min_mean_to_other_means = round(df["mean_to_other_means"].min(), 2)
    median_mean_to_other_means = round(df["mean_to_other_means"].median(), 2)

    # Set up matplotlib figure
    plt.figure(figsize=(16, 10))

    # Determine optimal bin count using different methods and take their average
    # For mean_distance
    md_data = df["mean_distance"].dropna()
    md_fd_bins = np.histogram_bin_edges(md_data, "fd")  # Freedman-Diaconis rule
    md_sturges_bins = np.histogram_bin_edges(md_data, "sturges")  # Sturges' rule
    md_scott_bins = np.histogram_bin_edges(md_data, "scott")  # Scott's rule

    # For mean_to_other_means
    motm_data = df["mean_to_other_means"].dropna()
    motm_fd_bins = np.histogram_bin_edges(motm_data, "fd")
    motm_sturges_bins = np.histogram_bin_edges(motm_data, "sturges")
    motm_scott_bins = np.histogram_bin_edges(motm_data, "scott")

    # Calculate optimal bin counts
    md_bins = max(len(md_fd_bins), len(md_sturges_bins), len(md_scott_bins)) - 1
    motm_bins = max(len(motm_fd_bins), len(motm_sturges_bins), len(motm_scott_bins)) - 1

    print(f"Optimal bin count for mean_distance: {md_bins}")
    print(f"Optimal bin count for mean_to_other_means: {motm_bins}")

    # Create the histograms with improved styling
    sns.histplot(
        df["mean_distance"],
        bins=md_bins,
        kde=True,
        color="#1E5A84",  # Royal Blue
        label="Intra-Subject Distances",
        stat="density",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        element="step",
    )
    sns.histplot(
        df["mean_to_other_means"],
        bins=motm_bins,
        kde=True,
        color="#DC143C",  # Crimson
        label="Inter-Subject Distances",
        stat="density",
        alpha=0.7,
        edgecolor="white",
        linewidth=0.5,
        element="step",
    )

    # Add a more descriptive title with custom font
    plt.title(
        "Distribution of Embedding Distances Within vs. Between Subjects",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # Improve axis labels
    plt.xlabel("Euclidean Distance", fontsize=14, fontweight="semibold")
    plt.ylabel("Density", fontsize=14, fontweight="semibold")

    # Improve legend
    plt.legend(fontsize=12, frameon=True, facecolor="white", edgecolor="lightgray")

    # Customize the spines
    for spine in plt.gca().spines.values():
        spine.set_color("lightgray")

    # Add grid but make it subtle
    plt.grid(True, alpha=0.3, linestyle="-", color="lightgray")

    # Fix x-axis ticks to start from the left edge
    ax = plt.gca()
    x_min, x_max = ax.get_xlim()
    ax.spines["bottom"].set_position(("data", 0))
    ax.xaxis.set_ticks_position("bottom")

    # Ensure tick marks start from axes
    plt.tick_params(axis="both", which="major", labelsize=12, direction="out")
    ax.set_xlim(left=0)  # Force x-axis to start at 0 if appropriate for your data

    # Ensure layout is tight
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the plot with svg format for high quality
    plt.savefig(filename, format="svg", dpi=300)
    print(f"Enhanced histogram saved as '{filename}'")


df = inspect_and_load_pickle(
    config["EMBEDDER"]["RESULT_PATH"], distance_to_centroids=False
)
