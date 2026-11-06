#!/usr/bin/env python3
"""Plot churn distribution."""

import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Visualize the distribution of churn classes."""
    plt.figure(figsize=(12, 8))

    counts = df['Churn'].value_counts()

    plt.bar(
        ['No', 'Yes'],
        [counts.get('No', 0), counts.get('Yes', 0)],
        color=['skyblue', 'salmon']
    )

    plt.title('Churn Distribution')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

    return None
