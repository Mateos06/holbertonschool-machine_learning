#!/usr/bin/env python3
"""Standardize numeric features using StandardScaler."""

from sklearn import preprocessing


def scale_numeric(df):
    """Scale MonthlyCharges and TotalCharges."""
    scaler = preprocessing.StandardScaler()

    columns = ["MonthlyCharges", "TotalCharges"]

    df[columns] = scaler.fit_transform(df[columns])

    return df
