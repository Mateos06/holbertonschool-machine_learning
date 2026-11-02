#!/usr/bin/env python3
"""
Plot churn distribution.
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Visualize the distribution of churn classes.
    """
    plt.figure(figsize=(12, 8))

    counts = df['Churn'].value_counts()

    counts.plot(
        kind='bar',
        color=['skyblue', 'salmon']
    )

    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.show()
