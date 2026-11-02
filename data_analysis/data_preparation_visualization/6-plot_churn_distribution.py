#!/usr/bin/env python3
"""
Visualize the distribution of churn classes.
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Plot the counts of customers by churn status.
    """
    plt.figure(figsize=(12, 8))

    counts = df['Churn'].value_counts()
    counts.plot(kind='bar', color=['skyblue', 'salmon'])

    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.show()
