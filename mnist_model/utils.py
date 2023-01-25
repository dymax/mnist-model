"""
The following utility functions are provided:
- de_serialise: This function converts a tf.data.Dataset in the form of (image, labels) to a list.
- normalize_pixels: This function normalizes the pixels.
- load_config_json: This function loads and reads data from a json file.
- plot_confusion_matrix: This function plots a confusion matrix.
- misclassified_rate: This function calculates the misclassified rate.
- plot_misclassified_rate: This function plots the misclassified rate for each class.
"""

import json
from typing import Tuple, Dict, List
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def de_serialise(ds: tf.data.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the tf.data.Data object of image and label to numpy array.
    :param ds: A tf.data.Data object slice from (image, label).
    :return: A tuple of:
            - x: a numpy array of input image.
            - y: a numpy array of labels.
    """
    x, y = tf.data.Dataset.get_single_element(ds.batch(len(ds)))
    x = x.numpy()  # image
    y = y.numpy()  # labels
    return x, y


def normalize_pixels(image: np.ndarray, label: int):
    """
    Normalizes images and convert to `float32`.
    """
    return tf.cast(image, tf.float32) / 255., label


def load_config_json(file_path: str) -> Dict:
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], plot_tag: str) -> None:
    """
    Plot the confusion matrix.
    :param y_true: A list target values.
    :param y_pred: A list of estimated targets as returned by the model.
    :param class_names: a list of class name
    :param plot_tag: a string tag that wee want to plot's title
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix in a beautiful manner
    _ = plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g');  # annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Prediction', fontsize=10)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize=8)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True', fontsize=10)
    ax.yaxis.set_ticklabels(class_names, fontsize=8)
    plt.yticks(rotation=0)

    plt.title(f'Confusion Matrix on {plot_tag}', fontsize=12)
    plt.show()


def misclassified_rate(y_true: List[int], y_pred: List[int]) -> List[float]:
    """
    Compute the misclassified rate for each class as:
      misclassified_rate = (FN + FP) / (TP + TN + FP +FN)
    :param y_true: A list target values.
    :param y_pred: A list of estimated targets as returned by the model.
    :return: a list that contains the misclassified rate for each class.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Get the total number of samples per class
    total_samples_per_class = cm.sum(axis=1)

    # Initialize the misclassification rate per class
    misclassification_rate_per_class = []

    for i in range(len(cm)):
        # Get the number of misclassified samples for class i
        misclassified_samples = 0
        for j in range(len(cm)):
            if i != j:
                misclassified_samples += cm[i][j]
        # Calculate the misclassification rate for class i
        misclassification_rate_per_class.append(misclassified_samples / total_samples_per_class[i])
    return misclassification_rate_per_class


def plot_misclassified_rate(y_true: List[int], y_pred: List[int], class_names: List[str], plot_tag: str) -> None:
    """
    Plot misclassified plot rate as a bar graph for each class.
    :param y_true: A list target values.
    :param y_pred: A list of estimated targets as returned by the model.
    :param class_names:
    :param plot_tag:
    :return:
    """
    # Compute misclassified rate
    misclassification_rate_per_class = misclassified_rate(y_true, y_pred)

    # Create a bar chart of the misclassification rate for each class
    _ = plt.figure(figsize=(10, 8))
    plt.bar(class_names, misclassification_rate_per_class)
    plt.xlabel('Classes')
    plt.ylabel('Misclassification Rate')
    plt.title(f'Misclassification Rate per Class on {plot_tag}')
    plt.xticks(rotation=45, fontsize=8)
    plt.show()



