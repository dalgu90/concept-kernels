import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
)
from sklearn.feature_selection import r_regression


def to_np_array(array):
    # Keep None as it is, and convert others into Numpy array
    if array is not None and not isinstance(array, np.ndarray):
        array = np.array(array)
    return array

def select_column(array, col):
    if array is None or col is None:
        return array
    elif array.ndim == 1:
        return array
    else:
        return array[:, col]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Metric:
    def __init__(self, params):
        self.params = params
        # Selecting prediction/label columns
        self.true_col = None
        self.pred_col = None
        if self.params and "column" in self.params:
            self.true_col = self.params.column
            self.pred_col = self.params.column
        elif self.params and "true_column" in self.params:
            self.true_col = self.params.true_column
            self.pred_col = self.params.pred_column

        # Filtering rows by label value
        self.filter_col = None
        self.filter_val = None
        if self.params and "filter_column" in self.params:
            self.filter_col = self.params.filter_column
            self.filter_val = self.params.filter_value

    def select_filter_column(self, y_true, y_pred):
        y_true_col = select_column(to_np_array(y_true), self.true_col)
        y_pred_col = select_column(to_np_array(y_pred), self.pred_col)
        if self.filter_col is not None:
            y_filter = select_column(to_np_array(y_true), self.filter_col)
            y_filter = y_filter == self.filter_val
            y_true_col = y_true_col[y_filter]
            y_pred_col = y_pred_col[y_filter]
        return y_true_col, y_pred_col


    def __call__(self, y_true, y_pred):
        # y_true_col = select_column(to_np_array(y_true), self.true_col)
        # y_pred_col = select_column(to_np_array(y_pred), self.pred_col)
        # if self.filter_col is not None:
            # y_filter = select_column(to_np_array(y_true), self.filter_col)
            # y_filter = y_filter == self.filter_val
            # y_true_col = y_true_col[y_filter]
            # y_pred_col = y_pred_col[y_filter]
        y_true_col, y_pred_col = self.select_filter_column(y_true, y_pred)
        return float(self.forward(y_true_col, y_pred_col))

    def forward(self, y_true, y_pred):
        raise NotImplementedError("This is the base metric class")


###########################################
# Classification Metrics
###########################################

class Accuracy(Metric):
    def forward(self, y_true, y_pred):
        if y_true.ndim == y_pred.ndim:  # Binary classification
            y_pred = (y_pred >= 0.0).astype(float)
        elif y_true.ndim + 1 == y_pred.ndim:  # Multi-class classification
            y_pred = y_pred.argmax(axis=-1).astype(float)
        return accuracy_score(y_true, y_pred)


class AveragePrecision(Metric):
    def forward(self, y_true, y_pred):
        return average_precision_score(y_true, sigmoid(y_pred))


class AUC(Metric):
    def forward(self, y_true, y_pred):
        return roc_auc_score(y_true, sigmoid(y_pred))


class F1Score(Metric):
    def __init__(self, params):
        super().__init__(params)
        if self.params and "average" in self.params:
            self.average = self.params.average
        else:
            self.average = None

    def forward(self, y_true, y_pred):
        if y_true.ndim == y_pred.ndim:  # Binary classification
            y_pred = (y_pred >= 0.0).astype(float)
            average = 'binary' if (self.average is None) else self.average
        elif y_true.ndim + 1 == y_pred.ndim:  # Multi-class classification
            y_pred = y_pred.argmax(axis=-1).astype(float)
            average = 'macro' if (self.average is None) else self.average
        return f1_score(y_true, y_pred, average=average)

###########################################
# Regression Metrics
###########################################

class MeanSquaredError(Metric):
    def forward(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)


class RMSE(MeanSquaredError):
    def forward(self, y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)


class MeanAbsoluteError(Metric):
    def forward(self, y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)


class PearsonR(Metric):
    def forward(self, y_true, y_pred):
        return np.corrcoef(y_true, y_pred)[0, 1].item()
        # return r_regression(y_pred[:, np.newaxis], y_true).item()


"""
Mean of the ratio of prediction to ground truth, assuming both are positive
"""
class MeanRatio(Metric):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, y_true, y_pred):
        return np.mean(np.abs(y_pred/y_true))
