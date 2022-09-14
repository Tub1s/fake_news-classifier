import copy
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.sparse import hstack
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from typing import List, Tuple

class TFIDFTransform:
    def __init__(self, headline_vectorizer, body_vectorizer):
        self.__headline_vectorizer = copy.copy(headline_vectorizer)
        self.__body_vectorizer = copy.copy(body_vectorizer)


    def fit_transform(self, df):
        headline, body = df["headline"], df["article_body"]

        X_headline = self.__headline_vectorizer.fit_transform(headline)
        X_body = self.__body_vectorizer.fit_transform(body)

        X = hstack((X_headline, X_body))
        return X

    def transform(self, df):
        headline, body = df["headline"], df["article_body"]

        X_headline = self.__headline_vectorizer.transform(headline)
        X_body = self.__body_vectorizer.transform(body)

        X = hstack((X_headline, X_body))
        return X

class FakeNewsClassifier:
    def __init__(self, relation_classifier: xgb.Booster, class_classifier: xgb.Booster,
                 tfidf_transform: TFIDFTransform):
        self.__relation_classifier = relation_classifier
        self.__class_classifier = class_classifier
        self.__tfidf_transform = tfidf_transform

    def __convert_to_int(self, y):
        return np.where(y < 0.5, 0, 1)
     
    def __transform_to_dmatrix(self, X, y):
        X = self.__tfidf_transform.transform(X)
        dmatrix = xgb.DMatrix(X, label=y)

        return dmatrix

    def __predict_relation(self, X, y):
        dmatrix = self.__transform_to_dmatrix(X,y)
        y_pred = self.__relation_classifier.predict(dmatrix)
        y_pred = self.__convert_to_int(y_pred)

        return y_pred

    def __predict_class(self, X, y):
        dmatrix = self.__transform_to_dmatrix(X,y)
        y_pred = self.__class_classifier.predict(dmatrix)

        return y_pred

    def __find_correct_predicitons(self, y_test, y_pred):
        correct_indices = []
        for i in range(len(y_pred)):
            if (y_pred[i] == y_test[i]) and y_pred[i] == 1:
                correct_indices.append(i)
    
        return correct_indices

    def __find_cls_matching_indices(self, y_org, y_test_indices, matching_indices):
        X_indices = y_org.iloc[matching_indices].index

        y_indices = []
        for idx in X_indices:
            y_indices.append(y_test_indices.index(idx))

        return X_indices, y_indices


    def predict_test(self, X_rel: pd.DataFrame, y_rel: pd.DataFrame, 
                     X_cls: pd.DataFrame, y_cls: pd.DataFrame, 
                     indices: List[int], y_org: pd.DataFrame) -> Tuple:
        rel_pred = self.__predict_relation(X_rel, y_rel)
        correct_indices = self.__find_correct_predicitons(y_rel, rel_pred)

        X_indices, y_indices = self.__find_cls_matching_indices(y_org, indices, correct_indices)
        X_cls = X_cls.loc[X_indices]
        y_cls = y_cls[y_indices]

        final_predictions = self.__predict_class(X_cls, y_cls)

        return final_predictions, y_cls

    @staticmethod
    def print_metrics(y_true: np.ndarray | List[int], y_pred: np.ndarray | List[int]) -> None:
        average = "binary" if len(np.unique(y_true)) == 2 else "macro"
        print("Precision = {}".format(precision_score(y_true, y_pred, average=average)))
        print("Recall = {}".format(recall_score(y_true, y_pred, average=average)))
        print("Accuracy = {}".format(accuracy_score(y_true, y_pred)))
        print("Balanced accuracy = {}".format(balanced_accuracy_score(y_true, y_pred)))

    @staticmethod
    def plot_cm(y_true: np.ndarray | List[int], y_pred: np.ndarray | List[int], labels: List[str]) -> None:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot()