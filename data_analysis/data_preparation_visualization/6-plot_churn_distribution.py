#!/usr/bin/env python3
"""Plot churn distribution."""

import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Visualize the distribution of churn classes."""
    plt.figure(figsize=(12, 8))

    counts = df["Churn"].value_counts()
    counts = counts.reindex(["No", "Yes"], fill_value=0)

    counts.plot(
        kind="bar",
        color=["skyblue", "salmon"],
        rot=0
    )

    plt.title("Churn Distribution")
    plt.ylabel("Count")
    plt.xlabel("")

    plt.show()
    return None
