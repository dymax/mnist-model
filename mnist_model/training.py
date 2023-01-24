import logging
import os
from pathlib import Path
from typing import Tuple, Dict, List, Union
from functools import partial
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load TensorFlow
import tensorflow as tf

# Load mlflow tracking tools
import mlflow

# Load hyperopt for hyperparameter search
from hyperopt import fmin, tpe, STATUS_OK, Trials
from hyperopt import hp

from mnist_model.data_loader import load_data
from mnist_model.model import SimpleModel
from mnist_model.utiles import normalize_pixels

logging.basicConfig(level=logging.INFO)


OUTPUT_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'outputs')
MODEL_PATH = os.path.join(OUTPUT_PATH, "trained_model", "model")

# from terminal run mlflow ui --backend-store-uri sqlite:///meas-energy-mlflow.db
mlflow.set_tracking_uri(f"sqlite:///{OUTPUT_PATH}/meas-energy-mlflow.db")

tf.get_logger().setLevel('ERROR')

SEED = 100
tf.random.set_seed(SEED)


def train_eval_pipeline() -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict[str, int]]:
    """
    Load train and test datasets from data loader and create a pipeline that contains:
      - normalise the data
      - fit the dataset in memory, cache it before shuffling for a better performance.
      - shuffle dataset.
      - Batch elements of the dataset after shuffling to get unique batches at each epoch.
    :return: prepared train and test dataset and data info.
    """
    # Load data from data loader
    dataset, data_info = load_data()
    # Normalise the train dataset
    ds_train = dataset["train"].map(normalize_pixels)
    # Normalise test dataset
    ds_test = dataset["test"].map(normalize_pixels)
    # Cache train data before shuffling for a better performance.
    ds_train = ds_train.cache()
    # Shuffle train dataset
    ds_train = ds_train.shuffle(data_info["num_labels"], seed=SEED)
    # Batch train dataset after shuffling to get unique batches at each epoch.
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Batch test dataset to get unique batches at each epoch.
    ds_test = ds_test.batch(128)
    # Cache test data before shuffling for a better performance.
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_train, ds_test, data_info


def training_job(save_model_path: str = MODEL_PATH,
                 dropout_rate: float = 0.3,
                 learning_rate: float = 0.001,
                 num_units: int = 128,
                 num_epochs: int = 10) -> Tuple[SimpleModel, Dict[str, List[float]]]:
    """
    Train and eval model.
    :param save_model_path: path to save the trained model.
    :param dropout_rate: drop out rate.
    :param learning_rate: learning rate.
    :param num_units: number of neurons/units of the network layer.
    :param num_epochs: number of epochs
    :return: A tuple:
             - model: A trained model.
             - history: history of the loss and accuracy for train and eval data
                        during model fitting.
    """
    ds_train, ds_test, data_info = train_eval_pipeline()
    image_shape = data_info['train']["shape"]
    num_labels = data_info["num_labels"]

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="../logs")
    # Define mlflow experiment name
    mlflow_experiment_name = f"model-training"
    # Setup mlflow experiment
    exp = mlflow.get_experiment_by_name(name=mlflow_experiment_name)
    if not exp:
        experiment_id = mlflow.create_experiment(name=mlflow_experiment_name,
                                                 artifact_location=f"{OUTPUT_PATH}/mlruns")
    else:
        experiment_id = exp.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=mlflow_experiment_name, nested=True):

        # Autolog the tensorflow model during the training
        mlflow.tensorflow.autolog(every_n_iter=2)

        model = SimpleModel(image_shape, dropout_rate, num_units, num_labels)
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # set the logits to False
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        # Fit the model and get the history
        history = model.fit(
            ds_train,
            epochs=num_epochs,
            validation_data=ds_test,
            verbose=1,
            callbacks=[tensorboard_callback]
        )
    # Evaluate the model on the test dataset
    metrics = model.evaluate(ds_test)

    # Print the evaluation results
    logging.info(f"Test Loss: {metrics[0]}")
    logging.info(f"Test Accuracy: {metrics[1]}")
    # Save model
    model.save(save_model_path)
    return model, history


def objective(params: Dict[str, Union[int, float]],
              ds_train: tf.data.Dataset,
              ds_test: tf.data.Dataset,
              num_epochs: int) -> Dict[str, Union[str, float]]:
    """
    Objective function that will be used to minimise the loss during the training process.
    :param params: a dictionary of parameters that will be used to compute the loss.
    :param ds_train: training datasett.
    :param ds_test: testing dataset
    :param num_epochs: number of epoch to train the model.
    :return: A data dictionary:
            - loss:  a float value that attempting to minimise
            - status: Status of completion; ok' for successful completion, and 'fail' in cases where the function turned
                      out to be undefined.
    """
    # Define mlflow experiment name
    mlflow_experiment_name = f"model-hyper-search"
    # Setup mlflow experiment
    exp = mlflow.get_experiment_by_name(name=mlflow_experiment_name)
    if not exp:
        experiment_id = mlflow.create_experiment(name=mlflow_experiment_name,
                                                 artifact_location=f"{OUTPUT_PATH}/mlruns")
    else:
        experiment_id = exp.experiment_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=mlflow_experiment_name, nested=True):
        # Autolog the tensorflow model during the training
        mlflow.tensorflow.autolog(every_n_iter=2)

        model = SimpleModel(image_shape=(28, 28),
                            dropout_rate=params['dropout_rate'],
                            num_units=params['num_units'],
                            num_labels=10)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )
        model.fit(ds_train,
                  epochs=num_epochs,
                  validation_data=ds_test,
                  batch_size=128,
                  verbose=1)

        # Get loss from eval model, the loss will be minimised by objective function
        eval_metrics = model.evaluate(ds_test)

    return {'loss': eval_metrics[0], 'status': STATUS_OK}


def run_hyper_search(max_eval: int, num_epochs: int):
    """
    Run hyperparameter search space to find the optimal set of parameters.
    :param max_eval: Maximum number of iteration to run the search space.
    :param num_epochs: number of epoch to train the model.
    :return: Result of search space
    """
    ds_train, ds_test, data_info = train_eval_pipeline()
    # Define the search space. This is only used for the purpose of the demo.
    # Only learning rate, dropout ratio and number of neurons considered as hyperparameter
    params = {'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.1)),
              'num_units': hp.quniform('num_units', 16, 256, 16),
              'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5)}
    # Create the objective function that wants to be minimised
    obj = partial(objective,
                  ds_train=ds_train,
                  ds_test=ds_test,
                  num_epochs=num_epochs)

    # Minimise the objective over the space
    result = fmin(
        fn=obj,
        space=params,
        algo=tpe.suggest,
        max_evals=max_eval,
        trials=Trials(),
        verbose=True)
    return result


if __name__ == "__main__":
    training_job(num_epochs=10)
    run_hyper_search(max_eval=10, num_epochs=10)
