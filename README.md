# Abstract
It's a Federated Neural Network ecossystem, where there are two components.
a) Federated Central: Responsible for ministering the primary model
b) Federated Train: A example of neural network train that will be federated to central.

## Federated Central
Is a Flask App that receives requests of the Federated Trainners with its models, and merge them with the primary model.
This component will have a XAI controllers responsible for check if a new federated train is able to be trusted.

## Federated Train
Is a simple MNIST neural network in trainnig, that will send your purchased weights through requests to the central.

## Running
```python3 -m venv venv```
```source venv/bin/activate```
```pip install -r requirements.txt```

- Federated Central:
  ```$cd FederatedCentral```
  ```$flask run```

- Federated Train:
  ```$cd FederatedTrain```
  ```$python3 -m main```
