import logging
import os
from typing import Dict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

logging.basicConfig(level=logging.INFO)


def load_data() -> Dict[str, tf.data.Dataset]:
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
    logging.info(f"Train size {len(x_train)}")
    logging.info(f"Test size {len(x_test)}")

    # Convert data to tf.data.Data object. Combining x_train and y_train as it would be easier to shuffle
    # the data before fitting to the model.
    train_dataset = tf.data.Dataset.from_tensor_slices({"x_train": x_train, "y_train": y_train})
    test_dataset = tf.data.Dataset.from_tensor_slices({"x_test": x_test, "y_test": y_test})

    return {"train": train_dataset, "test": test_dataset}


if __name__ == "__main__":
    dataset = load_data()
    for val in dataset["train"].take(1).as_numpy_iterator():
        print(val["x_train"].shape)
        print(val["y_train"])
