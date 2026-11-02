#!/usr/bin/env python3
"""Plot numeric feature distributions by churn."""

import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Compare numeric feature distributions by churn."""

    no_churn = df[df["Churn"] == "No"][col]
    yes_churn = df[df["Churn"] == "Yes"][col]

    plt.figure(figsize=(12, 8))

    plt.hist(
        [no_churn, yes_churn],
        bins=30,
        label=["No", "Yes"]
    )

    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title="Churn", loc="upper center")

    plt.show()
    return None
