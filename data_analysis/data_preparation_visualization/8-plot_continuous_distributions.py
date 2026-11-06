#!/usr/bin/env python3
"""Visualize distributions of continuous numerical features."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_continuous_distributions(df, columns_to_plot=None):
    """Plot histograms with KDE and boxplots."""

    if columns_to_plot is None:
        columns_to_plot = df.select_dtypes(
            include="number"
        ).columns.tolist()

    for col in columns_to_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: Histogram with KDE
        sns.histplot(
            df[col].dropna(),
            bins=30,
            stat="density",
            alpha=0.7,
            edgecolor="black",
            kde=False,
            ax=axes[0]
        )

        sns.kdeplot(
            df[col].dropna(),
            color="red",
            ax=axes[0]
        )

        axes[0].set_title(f"{col} Histogram + KDE")

        # Right: Boxplot
        sns.boxplot(
            y=df[col].dropna(),
            ax=axes[1]
        )

        axes[1].set_title(f"{col} Boxplot")

        plt.tight_layout()
        plt.show()

    return None
