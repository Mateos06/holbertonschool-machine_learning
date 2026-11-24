#!/usr/bin/env python3
"""Logistic Regression classifier."""

from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """Return an untrained Logistic Regression model."""
    model = linear_model.LogisticRegression(random_state=random_state)
    return model
