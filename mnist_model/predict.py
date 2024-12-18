"""
This script performs the following tasks:
- Loading the trained model: It loads the trained model that has been saved in the outputs folder.
- Making predictions: It makes predictions for both the train and test datasets.
- Generating plots: It generates confusion matrix and misclassified rate plots for both the train and test datasets.
"""

import os
from typing import List, Tuple
from pathlib import Path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from mnist_model.data_loader import convert_data_to_tf_dataset
from mnist_model.utils import normalize_pixels, de_serialise, load_config_json, plot_confusion_matrix, plot_misclassified_rate

OUTPUT_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'outputs')
CONFIG_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'configs')


def make_prediction(loaded_model_path: str, ds: tf.data.Dataset) -> Tuple[List[int], List[int]]:
    """
    Make prediction on the provided model and dataset.
    :param loaded_model_path: file path of the saved model.
    :param ds: dataset in the form of tf.data.Dataset object in the format (image, labels).
    :return: A tuple of:
            - y_true: a list target values.
            - y_pred: a list of estimated targets as returned by the model.
    """
    loaded_model = tf.keras.models.load_model(loaded_model_path)
    # Normalise the train dataset
    ds = ds.map(normalize_pixels)
    _, y_true = de_serialise(ds)
    # Create a batch
    ds = ds.batch(128)
    y_pred_proba = loaded_model.predict(ds)
    y_pred = [proba.argmax() for proba in y_pred_proba]
    return y_true, y_pred


if __name__ == "__main__":
    labels_to_int = load_config_json(os.path.join(CONFIG_PATH, 'labels.json'))
    # Get Class Labels
    class_names = list(labels_to_int.keys())
    model_path = os.path.join(OUTPUT_PATH, "trained_model", "model")

    # Load data from data loader
    dataset, data_info = convert_data_to_tf_dataset()
    # Make prediction on batch of data
    y_train, y_train_pred = make_prediction(loaded_model_path=model_path, ds=dataset["train"])
    y_test, y_test_pred = make_prediction(loaded_model_path=model_path, ds=dataset["test"])

    plot_confusion_matrix(y_train, y_train_pred, class_names, plot_tag='Train')
    plot_confusion_matrix(y_test, y_test_pred, class_names, plot_tag='Test')
    plot_misclassified_rate(y_train, y_train_pred, class_names, plot_tag='Train')
    plot_misclassified_rate(y_test, y_test_pred, class_names, plot_tag='Test')