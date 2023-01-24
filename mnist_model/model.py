import tensorflow as tf

SEED = 10
tf.random.set_seed(SEED)


class SimpleModel(tf.keras.Model):
    def __init__(self, image_shape, dropout_rate, num_units, num_labels):
        super().__init__()
        initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
        self.flatten = tf.keras.layers.Flatten(input_shape=image_shape)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate, seed=SEED)  # adding a dropout to minimize the risk of overfitting
        self.dense1 = tf.keras.layers.Dense(units=num_units, activation='relu', kernel_initializer=initializer)
        self.dense2 = tf.keras.layers.Dense(num_labels, kernel_initializer=initializer)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, input_tensor, training=False):
        x = self.flatten(input_tensor)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.softmax(x)
