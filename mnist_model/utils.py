import json
import os
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


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str]):

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

    plt.title('Confusion Matrix', fontsize=12)
    plt.show()


def misclassified_rate(y_true: List[int], y_pred: List[int]) -> List[float]:
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


def plot_misclassified_rate(y_true: List[int], y_pred: List[int], class_names: List[str]) -> None:
    # Compute misclassified rate
    misclassification_rate_per_class = misclassified_rate(y_true, y_pred)

    # Create a bar chart of the misclassification rate for each class
    _ = plt.figure(figsize=(10, 8))
    plt.bar(class_names, misclassification_rate_per_class)
    plt.xlabel('Classes')
    plt.ylabel('Misclassification Rate')
    plt.title('Misclassification Rate per Class')
    plt.xticks(rotation=45, fontsize=8)
    plt.show()



