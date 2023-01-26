
install:
	pip install -r requirement.txt

train:
	python -m mnist_model.training --option train

model-runer:
	python -m mnist_model.model_runer

search:
	python -m mnist_model.training --option search

train-search:
	python -m mnist_model.training --option all

predict:
	python -m mnist_model.predict





