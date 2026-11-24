#!/usr/bin/env python3
"""Lasso Regression with L1 regularization."""

from sklearn import linear_model


def lasso_regression(random_state):
    """Return an untrained Lasso Regression model."""
    model = linear_model.Lasso(random_state=random_state)
    return model
