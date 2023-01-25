# mnist-model
This repository contains the implementation of a Convolutional networks (2 layers of ConvNet used) to classify the fashion MNIST dataset. The code is structured into several files that handle different aspects of the project, such as data loading, model implementation, training, prediction, and logging.

## Code structure
[mnist_model/data_loader.py](mnist_model/data_loader.py): converts the MNIST dataset to a tf.data.Dataset object. <br>
[mnist_model/utils.py](mnist_model/utils.py): contains a set of utility functions for normalizing the dataset, generating confusion matrices, and misclassification rates, and generating plots. <br>
[mnist_model/model.py](mnist_model/model.py): contains a class for the implemented model. <br>
[mnist_model/training.py](mnist_model/training.py): contains scripts for training the model and performing hyperparameter search using Bayesian optimization and mlflow for tracking the training and tuning process. <br>
[mnist_model/predict.py](mnist_model/predict.py): loads the trained model and makes predictions and plots confusion matrix and miscalssified samples. <br>
[outputs/meas-energy-mlflow.db](outputs/meas-energy-mlflow.db): is a database containing information logged during training and hyperparameter search.. <br>
[outputs/trained_model](outputs/trained_model): directory contains the trained model. <br>
[configs/config_path.json](configs/config_path.json): contains path to data source.
[configs/labels.json](configs/labels.json): contains class names mapped to integer representations. <br>
[configs/config_hparams.json](configs/config_hparams.json): contains parameters used for training the model. <br>
[requirement.txt](requirement.txt): is a list of requirements that must be installed to run the provided code.

## Local setup 
- Setup virtual environment   
```commandline
python3.8 -m venv venv 
source venv/bin/activate 
```
- Install all dependencies 
```commandline
make install
```

## Setup Mlflow tracking service
- Open a terminal on your local machine and activate the virtual environment that was previously set up.
- Change the directory to the `outputs` folder where the `MLflow` tracking database is stored.
- Run the command mlflow ui to launch the MLflow tracking server, which will be accessible at the default port of 5000.
```commandline
mlflow ui --backend-store-uri sqlite:///meas-energy-mlflow.db
```
Note: we can check whether the port 5000 is free or not by running the following command
```commandline
lsof -i:5000
```

## Description of [configs/config_hparams.json](configs/config_hparams.json) 
Contains set of parameters to run the model
- `num_epochs`: number of epochs to train the model.
- `learning_rate`: learning rate of the optimiser.
- `dropout_rate`: dropout rate for the dropout layer.
- `batch_size`: batch size used to train the model.
- `max_eval`: number of iterations to perform the hyperparameter tuning process, used by hyperopt.
- `num_filter_layer_1`: number of filter for the Conv2D at the first layer.
- `num_filter_layer_2`: number of filter for the Conv2D at the second layer.
- `kernel_size_layers`: kernel size that has been used by the model for the Conv2D layers.

Note: The model only provides the hyperparameter search for a few parameters for the purpose of the demo. The following parameters have been considered for search:
- `num_filter_layer_1`
- `num_filter_layer_2`
- `dropout_rate`
- `learning_rate`

## [configs/config_path.json](configs/config_path.json) Set up
It should be set up by users according to their local machine to reflect the path to data source:
- `DATA_PATH`: path to get the mnist data source.


## Run [training.py](mnist_model/training.py)
The provided script can perform two tasks:
1. Train the model using the set of parameters defined in [configs/config_hparams.json](configs/config_hparams.json).
2. Perform a hyperparameter search using the Bayesian optimization method to find the optimal set of parameters. The specific parameter search space is defined in https://github.com/dymax/mnist-model/blob/b73599ef7b3faaacca46dbb7e858f754d91af4fc/mnist_model/training.py#L209.

To utilize the script, please take the following steps:
- Start __mlflow__ as the script is set up to use it for both training and hyperparameter search.
- Navigate to the `mnist-model` repository and activate the virtual environment.
- The script offers the ability to run both model training and hyperparameter search or either one separately through the --option flag:
 - Run only the hyperparameter search by using either `python -m mnist_model.training --option search` or `make search`.
 - Run only the model training by using either `python -m mnist_model.training --option train` or `make train`.
 - Run both the model training and hyperparameter search by using either `python -m mnist_model.training --option all` or `make train-search`.

## Run [predict.py](mnist_model/predict.py)
The provided script aims to plot the confusion matrix and misclassification rate for the train and test datasets. It loads the trained model from outputs/trained_model and uses it to make predictions on the train and test datasets. Then it calculates the confusion matrix and misclassification rate. <br>
To run the script, please take the following steps:
- Navigate to the `mnist-model` repository and activate the virtual environment.
- Run one of the following commands to visualize the model performance:
```commandline
make predict
```
```
python -m mnist_model.predict
```


## Run makefile 
The provided [Makefile](Makefile)  contains a set of command lines that can be used to more easily execute the provided python scripts. The `Makefile` includes the following commands:
- `make install` installs all dependencies.
- `make train` runs the [mnist_model/training.py](mnist_model/training.py) script in ___Only runs the model training___. <br>
- `make search:` runs the [mnist_model/training.py](mnist_model/training.py) script in ___Only runs the hyperparameter search___. <br>
- `make train-search:` runs the [mnist_model/training.py](mnist_model/training.py) script in ___Runs both model training and hyperparameter search___. <br>
- `make predict` runs the [mnist_model/predict.py](mnist_model/predict.py) script. <br>






