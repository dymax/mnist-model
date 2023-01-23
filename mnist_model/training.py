from typing import Tuple
from functools import partial
import tensorflow as tf

import mlflow
from hyperopt import fmin, tpe, STATUS_OK, Trials

from mnist_model.data_loader import load_data
from mnist_model.model import SimpleModel

# from terminal run mlflow ui --backend-store-uri sqlite:///meas-energy-mlflow.db
mflow.set_tracking_uri(f"sqlite:///meas-energy-mlflow.db")

SEED = 100
tf.random.set_seed(SEED)


def normalize_pixels(image, label):
    """
  Normalizes images and convert to `float32`.
  """
    return tf.cast(image, tf.float32) / 255., label


def prepare_dataset():
    dataset, data_info = load_data()
    ds_train = dataset["train"].map(normalize_pixels)
    ds_test = dataset["test"].map(normalize_pixels)
    return ds_train, ds_test, data_info


def split_dataset(ds_train, ds_test, num_training_samples):
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(num_training_samples, seed=SEED)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test


def training_job(dropout_rate: float = 0.3,
                 num_units: int = 128):
    ds_train, ds_test, data_info = prepare_dataset()
    num_training_samples = data_info['train']["num_samples"]
    ds_train, ds_test = split_dataset(ds_train, ds_test, num_training_samples)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs")
    # Call the model
    model = SimpleModel((28, 28), 0.3, 128, 10)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=1,
        callbacks=[tensorboard_callback]
    )



def run_hyper_search(max_eval: int = 100):
    obj = bj = partial(objective,
                  train_dataframe=train_dataframe,
                  target=target,
                  config_features=config_features,
                  num_split=num_split,
                  model_name=model_name)


if __name__ == "__main__":
    training_job()
