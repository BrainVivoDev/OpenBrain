import pandas as pd
import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def high_or_low_valence(x, thres):
    return "High" if x >= thres else "Low"


def high_or_low_arousal(x, thres):
    return "High" if x >= thres else "Low"


def label_emotion(row):
    if row["valence_label"] == "Low" and row["arousal_label"] == "Low":
        return "Sad"
    elif row["valence_label"] == "Low" and row["arousal_label"] == "High":
        return "Angry"
    elif row["valence_label"] == "High" and row["arousal_label"] == "Low":
        return "Calm"
    else:  # High Val, High Aro
        return "Excited"


def plot_emotion_tsne(X, y_valence, y_arousal, title="", perplex=5):

    # perplex = min(5, X.shape[0] - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplex)
    X_2d = tsne.fit_transform(X)

    merged_df = pd.DataFrame(
        {
            "tsne_x": X_2d[:, 0],
            "tsne_y": X_2d[:, 1],
            "Valence_mean": y_valence,
            "Arousal_mean": y_arousal,
        }
    )

    ############################
    #  DEFINE HIGH/LOW LABELING & EMOTION NAMES
    ############################
    valence_threshold = 4.0
    arousal_threshold = 4.0

    merged_df["valence_label"] = merged_df["Valence_mean"].apply(
        high_or_low_valence, thres=valence_threshold
    )
    merged_df["arousal_label"] = merged_df["Arousal_mean"].apply(
        high_or_low_arousal, thres=arousal_threshold
    )

    # Combine into 4 categories:
    # Low Val / Low Aro -> "Sad"
    # Low Val / High Aro -> "Angry"
    # High Val / Low Aro -> "Calm"
    # High Val / High Aro -> "Excited"

    merged_df["emotion_label"] = merged_df.apply(label_emotion, axis=1)

    ############################
    # PLOT T-SNE WITH DISCRETE 4-CATEGORY LABELS
    ############################
    # Example color mapping for the 4 combos
    category_colors = {
        "Sad": "blue",
        "Angry": "red",
        "Calm": "green",
        "Excited": "orange",
    }

    plt.figure(figsize=(8, 6))
    for emotion, color in category_colors.items():
        subset = merged_df[merged_df["emotion_label"] == emotion]
        plt.scatter(
            subset["tsne_x"], subset["tsne_y"], c=color, label=emotion, alpha=0.7
        )

    plt.title("t-SNE of " + title + " (4 Emotions)")

    plt.xlabel("t-SNE X")
    plt.ylabel("t-SNE Y")
    plt.legend()
    plt.tight_layout()
    plt.show()
