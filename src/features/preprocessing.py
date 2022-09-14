import numpy as np
import pandas as pd
from typing import Callable, Tuple

def data_preprocessor(X: pd.DataFrame, y: pd.DataFrame, encoder: Callable) -> Tuple:
    # Find indices for matching headline-body classification step
    y_cls_indices = y["stance"] != "unrelated"
    y_cls_indices = y_cls_indices[y_cls_indices].index

    # Create labels for binary classification (1: related, 0: unrelated)
    X_rel = X.copy
    y_rel = y.copy()
    y_rel["relation"] = y_rel.apply(lambda x: 1 if x["relation"] == "related" else 0, axis=1)
    y_rel = np.array(y_rel["relation"])

    # Create training dataset subset for related headline-body classification
    X_cls = X.loc[y_cls_indices]
    y_cls = y.loc[y_cls_indices]

    # Create labels for multiclass classification (agree, disagree, discuss)
    label_encoder = encoder
    y_cls = label_encoder.fit_transform(y_cls["stance"])

    return X_rel, X_cls, y_rel, y_cls