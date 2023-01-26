"""
This script serves as a runner for the "model.py" file, where the class "SimpleModel" is defined. It allows for
the parametrization of the model through the use of a config file. The script parses these
inputs and instantiates the "SimpleModel" class with the specified configuration.
"""
import ast
import logging
from pathlib import Path
import random
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

from mnist_model.model import SimpleModel
from mnist_model.utils import load_config_json
from mnist_model.training import train_eval_pipeline

# Set the random seed for tensorflow, numpy and random for consistency.
SEED = 100
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

logging.basicConfig(level=logging.INFO)

# Output path to store models
OUTPUT_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'outputs')
MODEL_PATH = os.path.join(OUTPUT_PATH, "trained_model", "model")
# Get params from config_params
CONFIG_PARAMS_PATH = os.path.join(Path(os.path.dirname(__file__)).parent, 'configs', 'config_hparams.json')
params = load_config_json(CONFIG_PARAMS_PATH)  # get the parameters

# Convert kernel_size of string to tuple of integers
params["kernel_size_layers"] = ast.literal_eval(params["kernel_size_layers"])

# Get and test sets
ds_train, ds_test, data_info = train_eval_pipeline(params["batch_size"])

# Build the model
model = SimpleModel(image_shape=data_info['train']["shape"],
                    num_filter_layer_1=params["num_filter_layer_1"],
                    num_filter_layer_2=params["num_filter_layer_1"],
                    kernel_size_layers=params["kernel_size_layers"],
                    dropout_rate=params["dropout_rate"],
                    num_labels=data_info["num_labels"])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  # set the logits to False
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
tensorboard_callback = [tf.keras.callbacks.TensorBoard(log_dir=f"{OUTPUT_PATH}/logs")]

# Fit the model
model.fit(ds_train,
          epochs=params["num_epochs"],
          validation_data=ds_test,
          batch_size=params["batch_size"],
          verbose=1,
          callbacks=tensorboard_callback)

# Evaluate the model on the test dataset
metrics = model.evaluate(ds_test)
logging.info(f"Test Loss: {metrics[0]}")
logging.info(f"Test Accuracy: {metrics[1]}")