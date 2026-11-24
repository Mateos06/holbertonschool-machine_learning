#!/usr/bin/env python3
"""Support Vector Machine classifier."""

from sklearn import svm


def get_SVM_model(name, random_state):
    """Return an untrained SVM classifier with the specified kernel."""
    if name not in ["linear", "poly", "rbf"]:
        raise ValueError("Unsupported kernel")

    model = svm.SVC(kernel=name, random_state=random_state)
    return model
