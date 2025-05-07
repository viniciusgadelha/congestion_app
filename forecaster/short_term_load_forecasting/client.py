# Libraries
import torch
import time
from collections import OrderedDict
import numpy as np
from typing import List, Tuple
from flwr.common import Metrics
import joblib

import flwr as fl
from .models.TCN import TCN

from omegaconf import DictConfig
from hydra.utils import instantiate
from .utils.utils import give_metrics

############################################# Centralized ###############################################

def train(model, train_dataset, epochs, learning_rate):
    """ Training function:
    After this function, the model will be trained.

        Parameters
    ----------
    model : torch.nn.Module
        Model that will be used to train.
    train_dataset : torch.data.utils.TensorDataset
        ´TensorDataset´ with inputs and targets.
    epochs : int
        Number of epochs to train the model.
        ..

    Output
    ------
    None
    """
    # Setup a loss function
    loss_fn = torch.nn.MSELoss()
    # Setup an optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # Bulding a training loop in PyTorch
    for epoch in range(epochs):
        model.train()  # train mode in PyTorch sets all parameters that require gradients to gradients.
        for batch in train_dataset:
            # Extract data and target in batch
            X, y = batch
            X = X.unsqueeze(2)
            # Train model
            y_pred = model(X)
            # Calculate the loss
            loss = loss_fn(y_pred.reshape(y_pred.shape[0], -1), y)
            # Optimizer zero grad
            optimizer.zero_grad()
            # Perform backpropagation on the loss with respect to the parameters of the model
            loss.backward()
            # Step the optimizer
            optimizer.step()


def test(model, test_dataset, train_dataset=None, epochs=None, learning_rate=None, user=None):
    loss_fn = torch.nn.MSELoss()
    val_losses = []
    mae_losses = []
    mape_losses = []
    smape_losses = []
    # Testing space after training
    model.eval()
    if user is not None:
        scaler = joblib.load('.cache/pt_y' + user + '.pkl')
    with torch.no_grad():
        for batch in test_dataset:
            X, y = batch
            X = X.unsqueeze(2)
            val_pred = model(X)
            val_loss = loss_fn(val_pred.reshape(val_pred.shape[0], -1), y)
            if user is not None:
                y_pred_descaled = scaler.inverse_transform(val_pred.numpy().reshape(val_pred.shape[0], val_pred.shape[2]))
                y_descaled = scaler.inverse_transform(y.numpy())
                mse = np.mean((y_pred_descaled - y_descaled) ** 2)
                mae, mape, smape = give_metrics(y_pred_descaled, y_descaled)
                val_losses.append(mse)
            else:
                mae, mape, smape = give_metrics(val_pred.reshape(val_pred.shape[0], -1).numpy(),
                                                y.numpy())
                val_losses.append(val_loss.detach().cpu().numpy())
            mae_losses.append(mae)
            mape_losses.append(mape)
            smape_losses.append(smape)
    val_loss = sum(val_losses) / len(val_losses)
    mae_loss = sum(mae_losses) / len(mae_losses)
    mape_loss = sum(mape_losses) / len(mape_losses)
    smape_loss = sum(smape_losses) / len(smape_losses)
    # TODO: borrar esta parte para agilizar calculos
    if train_dataset is not None:
        inicio = time.time()
        train(model, train_dataset, epochs, learning_rate)
        fin = time.time()
        tiempo_ejecucion = fin - inicio
        metrics = {"mae": mae_loss, "mape": mape_loss, "smape": smape_loss, "tiempo": tiempo_ejecucion}
    else:
        metrics = {"mae": mae_loss, "mape": mape_loss, "smape": smape_loss}
    return val_loss, metrics

def weighted_average_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    maes = [num_examples * m["mae"] for num_examples, m in metrics]
    mapes = [num_examples * m["mape"] for num_examples, m in metrics]
    smapes = [num_examples * m["smape"] for num_examples, m in metrics]
    tiempo = [num_examples * m["tiempo"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"mae": sum(maes) / sum(examples), "mape": sum(mapes) / sum(examples), "smape": sum(smapes) / sum(examples),
            # "tiempo": sum(tiempo) / sum(examples)}
            "tiempo": max(tiempo)}
############################################# Federated Learning ###############################################

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader, config):
        super().__init__()
        self.net = instantiate(config.model)
        self.train = train_loader
        self.test = test_loader
        self.device = config.config_fit.device
        self.local_epochs = config.config_fit.local_epochs
        self.learning_rate = config.config_fit.lr

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # parameters that will be inputed in the train
        # lr = config["lr"]
        train(self.net, self.train, self.local_epochs, self.learning_rate)
        return self.get_parameters(config), len(self.train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = test(self.net, self.test, self.train, self.local_epochs, self.learning_rate)
        return float(loss), len(self.test), metrics


def generate_client(ids, config: DictConfig):
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single client."""
        user_id_list = list(ids.keys())
        train_loader, test_loader = ids[user_id_list[int(cid)]]
        # Create a  single Flower client representing a single organization
        return FlowerClient(train_loader, test_loader, config)

    return client_fn