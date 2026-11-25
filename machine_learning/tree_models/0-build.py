#!/usr/bin/env python3
"""Module for building a decision tree classifier."""


from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """Build and return a decision tree classifier.

    Args:
        min_samples_leaf: Minimum samples required at a leaf node.
        min_samples_split: Minimum samples required to split a node.
        random_state: Seed used for reproducibility.

    Returns:
        An untrained DecisionTreeClassifier instance.
    """
    model = tree.DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    return model
