#!/usr/bin/env python3
"""Encode categorical features for machine learning."""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """Encode features and return fitted encoders."""
    df = df.copy()

    # Label encode Churn
    churn_le = preprocessing.LabelEncoder()
    df["Churn"] = churn_le.fit_transform(df["Churn"])

    # Ordinal encode binary columns
    binary_cols = [
        "Partner",
        "Dependents",
        "PaperlessBilling",
        "SeniorCitizen"
    ]

    binary_oe = preprocessing.OrdinalEncoder(
        categories=[["No", "Yes"]]
    )

    for col in binary_cols:
        df[[col]] = binary_oe.fit_transform(
            df[[col]]
        ).astype(int)

    # Ordinal encode TenureGroup
    tenure_oe = preprocessing.OrdinalEncoder()

    df[["TenureGroup"]] = tenure_oe.fit_transform(
        df[["TenureGroup"]]
    ).astype(int)

    # One-hot encode Contract and PaymentMethod
    df = pd.get_dummies(
        df,
        columns=["Contract", "PaymentMethod"],
        drop_first=True,
        dtype=int
    )

    return df, churn_le, binary_oe, tenure_oe
