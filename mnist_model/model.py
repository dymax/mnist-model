"""
Contains a class of the implemented model 'SimpleModel`.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

SEED = 10
tf.random.set_seed(SEED)


class SimpleModel(tf.keras.Model):
    def __init__(self, image_shape, num_filter_layer_1, num_filter_layer_2, kernel_size_layers, dropout_rate, num_labels):
        super(SimpleModel, self).__init__()
        self.num_filter1 = num_filter_layer_1
        self.kernel_size = kernel_size_layers
        self.num_filter2 = num_filter_layer_2
        self.image_shape = image_shape
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=self.image_shape),
                tf.keras.layers.Conv2D(self.num_filter1, kernel_size=self.kernel_size, activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(self.num_filter2, kernel_size=self.kernel_size , activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(self.dropout_rate),
                tf.keras.layers.Dense(self.num_labels, activation="softmax"),
            ]
        )

    def call(self, input_tensors, training=False):
        return self.model(input_tensors, training=training)

    def get_config(self):
        return {"image_shape": self.image_shape,
                "dropout_rate": self.dropout_rate,
                "num_units": self.num_units,
                "num_labels": self.num_labels}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
