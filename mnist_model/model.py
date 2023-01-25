"""
Contains a class of the implemented model 'SimpleModel`.
"""
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

SEED = 10
tf.random.set_seed(SEED)


class SimpleModel(tf.keras.Model):
    def __init__(self, image_shape, dropout_rate, num_units, num_labels):
        super(SimpleModel, self).__init__()
        self.image_shape = image_shape
        self.dropout_rate = dropout_rate
        self.num_units = num_units
        self.num_labels = num_labels
        initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
        self.flatten = tf.keras.layers.Flatten(input_shape=self.image_shape)
        self.dropout = tf.keras.layers.Dropout(rate=self.dropout_rate, seed=SEED)  # adding a dropout to minimize the risk of overfitting
        self.dense1 = tf.keras.layers.Dense(units=self.num_units, activation='relu', kernel_initializer=initializer)
        self.dense2 = tf.keras.layers.Dense(self.num_labels, kernel_initializer=initializer)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, input_tensors, training=False):
        x = self.flatten(input_tensors)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.softmax(x)

    def get_config(self):
        return {"image_shape": self.image_shape,
                "dropout_rate": self.dropout_rate,
                "num_units": self.num_units,
                "num_labels": self.num_labels}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def main():
    parser = argparse.ArgumentParser(description='SimpleModel')
    parser.add_argument('--image_shape', type=int, nargs='+', required=True, help='Shape of input images')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the model')
    parser.add_argument('--num_units', type=int, default=128, help='Number of units in the dense layer')
    parser.add_argument('--num_labels', type=int, required=True, help='Number of labels in the output')

    args = parser.parse_args()

    model = SimpleModel(args.image_shape, args.dropout_rate, args.num_units, args.num_labels)
    return model


if __name__ == "__main__":
    import pdb
    model = main()

