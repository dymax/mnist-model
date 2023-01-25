
install:
	pip install -r requirement.txt

only-train-model:
	python -m mnist_model.training --option train

only-hyper-search:
	python -m mnist_model.training --option search

train-search:
	python -m mnist_model.training --option all

predict:
	python -m mnist_model.predict





