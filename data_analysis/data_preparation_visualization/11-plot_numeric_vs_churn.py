#!/usr/bin/env python3
"""Plot numeric distributions by churn."""

import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Plot a numeric feature distribution grouped by churn."""
    no_data = df.loc[df["Churn"] == "No", col]
    yes_data = df.loc[df["Churn"] == "Yes", col]

    plt.figure(figsize=(12, 8))

    plt.hist(
        [no_data, yes_data],
        bins=30,
        label=["No", "Yes"]
    )

    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title="Churn", loc="upper center")

    plt.show()
    return None
