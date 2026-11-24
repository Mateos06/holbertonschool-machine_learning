#!/usr/bin/env python3
"""Ordinary Least Squares Linear Regression."""

from sklearn import linear_model


def Linear_Regression():
    """Return an untrained linear regression model."""
    model = linear_model.LinearRegression()
    return model
