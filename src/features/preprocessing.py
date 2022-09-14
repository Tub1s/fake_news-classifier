import numpy as np
import pandas as pd
from typing import Callable, Tuple

RELATION_MAPPINGS = {
    "agree": "related",
    "disagree": "related",
    "discuss": "related",
    "unrelated": "unrelated"
}


def df_preprocessor(body_df: pd.DataFrame, stances_df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic clean-up on both inupt DataFrames, merges them into one dataset and finally splits that set into feature and label sets.

    Args:
        body_df (pd.DataFrame): DataFrame containing data about article bodies
        stances_df (pd.DataFrame): DataFrame containing data about article stances

    Returns:
        pd.DataFrame: merged dataset
    """
    body_df = body_df.rename(columns={"Body ID": "body_id", "articleBody": "article_body"})
    stances_df = stances_df.rename(columns={"Body ID": "body_id"})
    stances_df.columns = stances_df.columns.str.lower()

    df = pd.merge(body_df, stances_df, on="body_id")

    df["relation"] = df.apply(lambda row: RELATION_MAPPINGS[row["stance"]], axis=1)
    df = df.drop(columns=["body_id"])

    return df


def dataset_preprocessor(X: pd.DataFrame, y: pd.DataFrame, encoder: Callable) -> Tuple:
    """
    Perofrms preprocessing on full DataFrame and returns feature-label DataFrame pairs for both stages of classification.
    Additionally it returns list of indices used to generate stance classification dataset.

    Args:
        X (pd.DataFrame): DataFrame containing headlines and article bodies
        y (pd.DataFrame): DataFrame containing labels (relations and stances)
        encoder (Callable): Label encoder for stance labels

    Returns:
        Tuple: feature-label DataFrame pairs (for relation and classification problems), indices used to generate stance classification set
    """    

    # Find indices for matching headline-body classification step
    cls_indices = y["stance"] != "unrelated"
    cls_indices = cls_indices[cls_indices].index

    # Create labels for binary classification (1: related, 0: unrelated)
    X_rel = X.copy()
    y_rel = y.copy()
    y_rel["relation"] = y_rel.apply(lambda x: 1 if x["relation"] == "related" else 0, axis=1)
    y_rel = np.array(y_rel["relation"])

    # Create training dataset subset for related headline-body classification
    X_cls = X.loc[cls_indices]
    y_cls = y.loc[cls_indices]

    # Create labels for multiclass classification (agree, disagree, discuss)
    label_encoder = encoder
    y_cls = label_encoder.fit_transform(y_cls["stance"])

    return X_rel, X_cls, y_rel, y_cls, list(cls_indices)
