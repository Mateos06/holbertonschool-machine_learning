#!/usr/bin/env python3
"""Create new features from customer data."""

import pandas as pd


def create_features(df):
    """Create NumServices and TenureGroup features."""

    service_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    # Count services marked as Yes
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)

    # Include InternetService (DSL or Fiber optic)
    df["NumServices"] += df["InternetService"].isin(
        ["DSL", "Fiber optic"]
    ).astype(int)

    # Create tenure groups
    bins = [0, 12, 24, 48, 60, float("inf")]
    labels = ["0-12", "13-24", "25-48", "49-60", "60+"]

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        right=True
    )

    # Drop original columns
    df.drop(
        columns=service_cols + ["InternetService", "tenure"],
        inplace=True
    )

    return df
