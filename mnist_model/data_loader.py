import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)
    print(y_train.shape)
    print(y_train)
    print(x_test.shape)
    print(y_test.shape)


if __name__ == "__main__":
    load_data()

