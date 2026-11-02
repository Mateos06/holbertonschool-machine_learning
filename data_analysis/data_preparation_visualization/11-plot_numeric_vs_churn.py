#!/usr/bin/env python3
"""Plot numeric distributions by churn."""

import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """Compare numeric feature distributions by churn."""
    plt.figure(figsize=(12, 8))

    plt.hist(
        df[df['Churn'] == 'Yes'][col],
        bins=30,
        alpha=0.5,
        label='Yes'
    )

    plt.hist(
        df[df['Churn'] == 'No'][col],
        bins=30,
        alpha=0.5,
        label='No'
    )

    plt.title(f'{col} Distribution by Churn')
    plt.xlabel(col)
    plt.legend(title='Churn')

    plt.show()
    return None