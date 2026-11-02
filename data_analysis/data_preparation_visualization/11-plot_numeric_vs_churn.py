#!/usr/bin/env python3
"""Plot numeric distributions by churn."""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_numeric_vs_churn(df, col):
    """Compare numeric distributions by churn."""
    plt.figure(figsize=(12, 8))

    sns.histplot(
        data=df,
        x=col,
        hue='Churn',
        bins=30,
        multiple='layer',
        alpha=0.5
    )

    plt.title(f'{col} Distribution by Churn')
    plt.xlabel(col)

    plt.show()
    return None