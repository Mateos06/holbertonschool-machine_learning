#!/usr/bin/env python3
"""Encode categorical features for machine learning."""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """Encode features and return fitted encoders."""
    df = df.copy()

    # 1. Label encode Churn
    churn_le = preprocessing.LabelEncoder()
    df["Churn"] = churn_le.fit_transform(df["Churn"])

    # 2. Ordinal encode binary columns
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

    # 3. Ordinal encode TenureGroup
    tenure_oe = preprocessing.OrdinalEncoder(
        categories=[sorted(df["TenureGroup"].unique())]
    )

    df[["TenureGroup"]] = tenure_oe.fit_transform(
        df[["TenureGroup"]]
    ).astype(int)

    # 4. One-hot encode categorical columns
    df = pd.get_dummies(
        df,
        columns=["Contract", "PaymentMethod"],
        drop_first=True,
        dtype=int
    )

    return df, churn_le, binary_oe, tenure_oe
