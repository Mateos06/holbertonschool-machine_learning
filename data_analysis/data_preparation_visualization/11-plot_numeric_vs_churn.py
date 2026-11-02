#!/usr/bin/env python3
"""Plot numeric feature distributions by churn."""

import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Plot numeric feature distributions for each churn category."""
    plt.figure(figsize=(12, 8))

    plt.hist(
        [
            df[df["Churn"] == "No"][col],
            df[df["Churn"] == "Yes"][col]
        ],
        bins=30,
        label=["No", "Yes"],
        color=["tab:blue", "tab:orange"]
    )

    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title="Churn", loc="upper center")

    plt.show()
