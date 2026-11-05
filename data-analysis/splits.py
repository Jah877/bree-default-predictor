"""Single reusable, reproducible split of applicants with observed outcomes."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(frame: pd.DataFrame, settings: dict) -> dict[str, pd.DataFrame]:
    data = settings["data"]
    ratios = settings["split"]
    train_f = float(ratios["train_fraction"])
    val_f = float(ratios["validation_fraction"])
    test_f = float(ratios["test_fraction"])
    if abs(train_f + val_f + test_f - 1.0) > 1e-8:
        raise ValueError("Train, validation and test fractions must sum to 1.")
    observed = frame.loc[
        frame[data["target"]].isin([data["positive_label"], data["negative_label"]])
    ].copy()
    random_state = int(settings["project"]["random_state"])
    train_and_val, test = train_test_split(
        observed, test_size=test_f, stratify=observed[data["target"]],
        random_state=random_state,
    )
    train, validation = train_test_split(
        train_and_val, test_size=val_f / (train_f + val_f),
        stratify=train_and_val[data["target"]], random_state=random_state,
    )
    return {"train": train.copy(), "validation": validation.copy(),
            "test": test.copy()}
