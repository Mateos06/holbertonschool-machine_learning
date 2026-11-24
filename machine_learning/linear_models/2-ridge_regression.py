#!/usr/bin/env python3
"""Ridge Regression with L2 regularization."""

from sklearn import linear_model


def ridge_regression(random_state):
    """Return an untrained Ridge Regression model."""
    model = linear_model.Ridge(random_state=random_state)
    return model
