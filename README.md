# mnist-model
This repo contains the implementation of a shallow network to classify the fashion MNIST dataset.

## Code structure
[mnist_model/data_loader.py](mnist_model/data_loader.py): a script that convert the mnist dataset to tf.data.Dataset object. <br>
[mnist_model/utils.py](mnist_model/utils.py): contains set of utility functions for normalising dataset, confusion matrix, misclassification-rate  and generate some plots. <br>
[mnist_model/model.py](mnist_model/model.py): contains a class of implemented model. <br>
[mnist_model/training.py](mnist_model/training.py): contains scripts to train a model and perform hyperparameter search using bayesian optimisation [hyperopt](https://github.com/hyperopt/hyperopt). Note, hyperparameter search is not comprehensive, only used for the demo. It's been integrated with mlflow to track the training and tuning process of the model. <br>
[mnist_model/predict.py](mnist_model/predict.py): script to load the trained model and make prediction and plots confusion matrix and miscalssified samples. <br>
[outputs/meas-energy-mlflow.db](outputs/meas-energy-mlflow.db): a database that contains the information that logs during the training and hyperparameters search. <br>
[outputs/trained_model](outputs/trained_model): contains the trained model. <br>
[configs/labels.json](configs/labels.json): contains the class names that is mapped to integer representation. <br>
[configs/config_hparams.json](configs/config_hparams.json): contains parameters that has been used to train the model. The parameters onlyy used for the demo purpose here. <br>
[requirement.txt](requirement.txt): list of requirement that shall be installed in order to run the provided codes.

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
- Open a terminal from you local machine and activate the virtual environment that has been already setup from `Local setup` step.
- change the directory to `outputs` folder where the mlflow tracking database has been stored.
- Run the following command to launch the mlflow uri. On default it runs as port 5000 
```commandline
mlflow ui --backend-store-uri sqlite:///meas-energy-mlflow.db
```
Note: we can check whether the port 5000 is free or not by running the following command
```commandline
lsof -i:5000
```
## Run [training.py](mnist_model/training.py)
The provided script can intend to provide two tasks
1. train the model on the set of parameters that can be provided at [configs/config_hparams.json]. 
2. perform hyperparameter search on the set of parameters that has defined https://github.com/dymax/mnist-model/blob/b73599ef7b3faaacca46dbb7e858f754d91af4fc/mnist_model/training.py#L209. The hyperparameter search uses bayesian optimisation method to search spaces for the optimal parameters. <br> </br>

To run the script please follow the following steps:
- Ensure the ___mlflow___ is running as the training and hyperparameter search has been set up with mlflow.
- cd directory to `mnist-model` repo and activate the virtual environment.
- The model training and hyperparameter search can be either run together or separately according to selected value for `--option` which accept three parameters:
  - Only runs the hyperparameter search by
    - Either `python -m mnist_model.training --option search` or `make search` 
  - Only runs the mode training
    - Either `python -m mnist_model.training --option train` or `make train`
  - Runs both model training and hyperparameter search by 
    - Either `python -m mnist_model.training --option all` or `make train-search`

## Run [predict.py](mnist_model/predict.py)
plot the confusion matrix and misclassified rate for eeach train and test dataset. The provided script loads the trained model from [outputs/trained_model](outputs/trained_model) and take predictions of the model from train and test dataset and then calculate the confusion matrix and misclassified rate . <br>
To run the script please follow the following steps:
- cd directory to `mnist-model` repo and activate the virtual environment.
- run either of the following command to visualise the model performance:
```option 1
make predict
```
```option 2
python -m mnist_model.predict
```

## Run makefile 
The provided [Makefile](Makefile) contains set of command lines that can be used to execute the python provided scripts in more convenient way. The provided `Makefile` contains:
- `make install` install all dependencies.
- `make train` execute the [mnist_model/training.py](mnist_model/training.py) file on the mode on ___Only runs the model training___. <br>
- `make search:` execute the [mnist_model/training.py](mnist_model/training.py) file on the mode on ___Only runs the hyperparameter search___. <br>
- `make train-search:` execute the [mnist_model/training.py](mnist_model/training.py) file on the mode on ___Runs both model training and hyperparameter search___. <br>
- `make predict` execute the [mnist_model/predict.py](mnist_model/predict.py). <br>






