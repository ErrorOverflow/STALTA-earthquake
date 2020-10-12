import pandas as pd
import h5py
import numpy as np
import obspy
from keras import backend as K
import matplotlib.pyplot as plt
from obspy.signal.trigger import recursive_sta_lta, classic_sta_lta, trigger_onset, ar_pick, recursive_sta_lta

def f1(y_true, y_pred):
    """

    Calculate F1-score.

    Parameters
    ----------
    y_true : 1D array
        Ground truth labels.

    y_pred : 1D array
        Predicted labels.

    Returns
    -------
    f1 : float
        Calculated F1-score.

    """

    def recall(y_true, y_pred):
        'Recall metric. Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of how many relevant items are selected.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        'Precision metric. Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of how many selected items are relevant.'

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

f1([1], [1])
