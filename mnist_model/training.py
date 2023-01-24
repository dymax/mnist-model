import os
from pathlib import Path
from typing import Tuple
from functools import partial
import tensorflow as tf
import numpy as np


import mlflow
from hyperopt import fmin, tpe, STATUS_OK, Trials
from hyperopt import hp
from hyperopt.pyll import scope

from mnist_model.data_loader import load_data
from mnist_model.model import SimpleModel

OUTPUT_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'outputs')


# from terminal run mlflow ui --backend-store-uri sqlite:///meas-energy-mlflow.db
mlflow.set_tracking_uri(f"sqlite:///{OUTPUT_PATH}/meas-energy-mlflow.db")

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

    history = model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
        verbose=1,
        callbacks=[tensorboard_callback]
    )
    # Evaluate the model on the test dataset
    metrics = model.evaluate(ds_test)
    # Print the evaluation results
    print("Test Loss:", metrics[0])
    print("Test Accuracy:", metrics[1])


def get_train_test_data():
    ds_train, ds_test, data_info = prepare_dataset()
    num_training_samples = data_info['train']["num_samples"]
    ds_train, ds_test = split_dataset(ds_train, ds_test, num_training_samples)
    return ds_train, ds_test


def objective(params,
              ds_train,
              ds_test):
    # define mlflow experiment name
    mlflow_experiment_name = f"model-hyper-search"
    # setup mlflow experiment
    mlflow.set_experiment(mlflow_experiment_name)
    with mlflow.start_run(run_name=mlflow_experiment_name, nested=True):
        model = SimpleModel((28, 28), params['dropout_rate'], params['num_units'], 10)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        model.fit(ds_train,
                  epochs=2,
                  validation_data=ds_test,
                  verbose=1)

        # log parameters from search space into mlflow
        mlflow.log_params(params)

        train_metrics = model.evaluate(ds_train)
        mlflow.log_metric("Train Accuracy", train_metrics[1])
        mlflow.log_metric("Train Loss", train_metrics[0])

        eval_metrics = model.evaluate(ds_test)
        mlflow.log_metric("Val Accuracy", eval_metrics[1])
        mlflow.log_metric("Val Loss", eval_metrics[0])

    return {'loss': eval_metrics[0], 'status': STATUS_OK}


def de_serialise(ds):
    def extract_x_y(x, y):
        return x, y
    ds_train = ds.map(extract_x_y)
    for x_i, y_i in ds_train:
        x = x_i
        y = y_i
    return x, y


def run_hyper_search(max_eval: int = 10):
    ds_train, ds_test = get_train_test_data()
    # Define the search space
    params = {'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
               'num_units': hp.quniform('num_units', 16, 256, 16),
               'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5)}
    x_train, y_train = de_serialise(ds_train)
    x_test, y_test = de_serialise(ds_test)
    obj = partial(objective,
                  ds_train=ds_train,
                  ds_test=ds_test)

    # minimize the objective over the space
    result = fmin(
        fn=obj,
        space=params,
        algo=tpe.suggest,
        max_evals=max_eval,
        trials=Trials(),
        verbose=True)
    return result


if __name__ == "__main__":
    #training_job()
    run_hyper_search()
