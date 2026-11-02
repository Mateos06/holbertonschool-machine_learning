#!/usr/bin/env python3
"""Visualize churn rates by categorical feature."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_categorical_vs_churn(df, col):
    """Plot churn rate for each category of a column."""
    churn_rate = df.groupby(col)["Churn"].apply(
        lambda x: (x == "Yes").mean()
    )

    plt.figure(figsize=(12, 8))

    sns.barplot(x=churn_rate.index, y=churn_rate.values)

    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)

    plt.show()

