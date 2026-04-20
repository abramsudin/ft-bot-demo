# ============================================================
# pipeline/loader.py
#
# Loads the training data and churn labels into a single
# DataFrame.  This is the only file that touches disk for
# raw data — everything downstream receives the DataFrame.
#
# Public API:
#   load_data(train_path, labels_path) -> pd.DataFrame
#
# Returns a DataFrame with:
#   - all original feature columns
#   - "churn" column (int, 0/1) appended
# ============================================================

import os
import numpy as np
import pandas as pd


def load_data(train_path: str, labels_path: str) -> pd.DataFrame:
    """
    Read train CSV and labels CSV, merge on position, return df.

    Parameters
    ----------
    train_path   : path to the feature CSV  (e.g. data/train (6).csv)
    labels_path  : path to the labels CSV   (e.g. data/train_churn_labels.csv)

    Returns
    -------
    pd.DataFrame with all original columns plus integer "churn" column.

    Raises
    ------
    FileNotFoundError if either path does not exist.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df     = pd.read_csv(train_path)
    labels = pd.read_csv(labels_path)

    df = df.copy()
    df["churn"] = (labels["Label"] == 1).astype(int)

    return df
