import logging
import os
from typing import Dict, Tuple
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

logging.basicConfig(level=logging.INFO)


def load_data() -> Tuple[Dict[str, tf.data.Dataset], Dict[str, int]]:
    """
    Load MNIST dataset and convert to tf.data.Data object.
    :return: A data dictionary of:
            - train: a collection of x_train and y_train as tf.data.Data object.
            - test: a collection of x_test and y_test as tf.data.Data object.
    """
    # Load mnist dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # print the number of samples for each train and test
    logging.info(f"Train size {x_train.shape}")
    logging.info(f"Test size {x_test.shape}")

    data_info = {"train": {"shape": x_train.shape[1:], "num_samples": x_train.shape[0]},
                 "test":  {"shape": x_test.shape[1:], "num_samples": x_test.shape[0]},
                 "num_labels": len(set(y_train))
                }

    # Convert data to tf.data.Data object. Combining x_train and y_train as it would be easier to shuffle
    # the data before fitting to the model.
    x_train = tf.data.Dataset.from_tensor_slices(x_train)
    y_train = tf.data.Dataset.from_tensor_slices(y_train)
    train_dataset = tf.data.Dataset.zip((x_train, y_train))

    x_test = tf.data.Dataset.from_tensor_slices(x_test)
    y_test = tf.data.Dataset.from_tensor_slices(y_test)
    test_dataset = tf.data.Dataset.zip((x_test, y_test))

    return {"train": train_dataset, "test": test_dataset}, data_info


if __name__ == "__main__":
    dataset, data_info = load_data()
    print(data_info)
    for val in dataset["train"].take(1).as_numpy_iterator():
        x, y = val
        print(x.shape)
        print(y)
