import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(methods, metrics, title):
    """
    Plots a comparison of various metrics for different methods.

    Parameters:
      methods (list): A list of method names.
      metrics (dict): A dictionary where keys are metric names and values are lists of metric scores
                      for each method (each list must be the same length as methods).
      title (str): The title for the plot (used for the figure's main title and the saved file name).
    """
    # Number of metrics to plot
    num_metrics = len(metrics)
    cols = 2
    rows = (num_metrics + cols - 1) // cols  # compute the required rows

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()  # flatten the array to simplify indexing

    # Iterate through each metric and create a bar chart
    for i, (metric, values) in enumerate(metrics.items()):
        # Ensure that the metric values align with the methods list
        if len(values) != len(methods):
            raise ValueError(
                f"Metric '{metric}' has {len(values)} values; expected {len(methods)} values."
            )

        ax = axes[i]
        x = np.arange(len(methods))  # create positions for each method
        width = 0.6  # width of each bar

        # Create the bar plot for this metric
        bars = ax.bar(x, values, width, color="skyblue", edgecolor="black")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha="right")

        # Annotate each bar with its value
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.grid(True, linestyle="--", alpha=0.5)

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add a main title for the figure
    fig.suptitle(f"{title}: comparison of Metrics", fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(f"{title}_metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()
