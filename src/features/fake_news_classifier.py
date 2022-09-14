import copy
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from typing import List, Tuple

class TFIDFTransform:
    """
    TFIDFTransform class provides custom TF-IDF based data vectorizer for fake news classification problem.
    """
    def __init__(self, headline_vectorizer, body_vectorizer):
        self.__headline_vectorizer = copy.copy(headline_vectorizer)
        self.__body_vectorizer = copy.copy(body_vectorizer)


    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Fit headline and body vectorizers main training set
        """
        headline, body = df["headline"], df["article_body"]

        X_headline = self.__headline_vectorizer.fit_transform(headline)
        X_body = self.__body_vectorizer.fit_transform(body)

        X = hstack((X_headline, X_body))
        return X

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Vectorize train (subset) and test data
        """
        headline, body = df["headline"], df["article_body"]

        X_headline = self.__headline_vectorizer.transform(headline)
        X_body = self.__body_vectorizer.transform(body)

        X = hstack((X_headline, X_body))
        return X


class FakeNewsClassifier:
    """
    FakeNewsClassifier provides custom pipeline for fake news classification problem.
    """
    def __init__(self, relation_classifier: xgb.Booster, stance_classifier: xgb.Booster,
                 tfidf_transform: TFIDFTransform):
        self.__relation_classifier = relation_classifier
        self.__stance_classifier = stance_classifier
        self.__tfidf_transform = tfidf_transform

    def __convert_to_int(self, y):
        """ Returns floating point predictions converted to integers """
        return np.where(y < 0.5, 0, 1)
     
    def __transform_to_dmatrix(self, X, y):
        """ Vectorizes data and builds DMatrix """
        X = self.__tfidf_transform.transform(X)
        dmatrix = xgb.DMatrix(X, label=y)

        return dmatrix

    def __predict_relation(self, X, y):
        """ Returns predicted relation between headline and body combinations """
        dmatrix = self.__transform_to_dmatrix(X,y)
        y_pred = self.__relation_classifier.predict(dmatrix)
        y_pred = self.__convert_to_int(y_pred)

        return y_pred

    def __predict_stance(self, X, y):
        """ Returns predicted stance of headline and body combinations """
        dmatrix = self.__transform_to_dmatrix(X,y)
        y_pred = self.__stance_classifier.predict(dmatrix)

        return y_pred

    def __find_correct_predicitons(self, y_test, y_pred):
        """ Returns list of indices of correctly predicted labels """
        correct_indices = []
        for i in range(len(y_pred)):
            if (y_pred[i] == y_test[i]) and y_pred[i] == 1:
                correct_indices.append(i)
    
        return correct_indices

    # TODO: Modify pipeline so that it doesn't rely on original DataFrame
    def __find_cls_matching_indices(self, y_org, y_test_indices, matching_indices):
        """ Returns lists of original rows indices that were correctly predicted """
        X_indices = y_org.iloc[matching_indices].index

        y_indices = []
        for idx in X_indices:
            y_indices.append(y_test_indices.index(idx))

        return X_indices, y_indices


    def predict_test(self, X_rel: pd.DataFrame, y_rel: np.ndarray, 
                     X_cls: pd.DataFrame, y_cls: np.ndarray, 
                     indices: List[int], y_org: pd.DataFrame) -> Tuple:
        """
        Predict stance of headline-body combination on test dataset.

        Args:
            X_rel (pd.DataFrame): DataFrame of features for relation classification
            y_rel (np.ndarray): Array of labels for relation classification
            X_cls (pd.DataFrame): DataFrame of features for stance classification
            y_cls (np.ndarray): Array of labels for stance classification
            indices (List[int]): List of indices from original DataFrame that are used only in stance classification step
            y_org (pd.DataFrame): DataFrame of original pre-transformation labels; used to access list of indices

        Returns:
            Tuple: predictions of stance and matching subset of ground truths
        """

        rel_pred = self.__predict_relation(X_rel, y_rel)
        correct_indices = self.__find_correct_predicitons(y_rel, rel_pred)

        X_indices, y_indices = self.__find_cls_matching_indices(y_org, indices, correct_indices)
        X_cls = X_cls.loc[X_indices]
        y_cls = y_cls[y_indices]

        final_predictions = self.__predict_stance(X_cls, y_cls)

        return final_predictions, y_cls

    @staticmethod
    def print_metrics(y_true: np.ndarray | List[int], y_pred: np.ndarray | List[int]) -> None:
        """ Prints basic metrics """
        average = "binary" if len(np.unique(y_true)) == 2 else "macro"
        print("Precision = {}".format(precision_score(y_true, y_pred, average=average)))
        print("Recall = {}".format(recall_score(y_true, y_pred, average=average)))
        print("Accuracy = {}".format(accuracy_score(y_true, y_pred)))
        print("Balanced accuracy = {}".format(balanced_accuracy_score(y_true, y_pred)))

    @staticmethod
    def plot_cm(y_true: np.ndarray | List[int], y_pred: np.ndarray | List[int], labels: List[str]) -> None:
        """ Plots simple confusion matrix """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot()
