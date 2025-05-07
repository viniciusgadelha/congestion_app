############################ CENTRALIZED APPROACH ############################
# libraries
import pandas as pd
import time
from tqdm import tqdm
import torch
import argparse
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
from itertools import chain
import matplotlib.pyplot as plt
# from openpyxl import load_workbook
from .client import test
import numpy as np
from .models.TCN import TCN, LSTMMIMO
from .utils.utils import mean_metrics, plotear_res_final
from hydra.utils import instantiate


# Use graphs edition
from .graficas import *
plotear = False

"""
Centralized scenario where the server trains the model with users' data. Communication between users does not exist.
Only with the central server.
The input parameters of this files are the number of epochs and the model used. Number of epochs is 300 by default,
but can be any other integer. On the other hand, the model is TCN by default, but can be LSTM. To run the model, use
the following command:
    python centralized_scenario.py --model TCN --epochs 300
    python centralized_scenario.py --model LSTM --epochs 300
    
Functions used in this code:
treat_data() -> tuple: Join all the data from the users as a sequential dataset.
train(model, train_loader, epochs, device): Training function adapted to the centralized scenario.
main(model, epochs): Main function to run the centralized scenario.
"""

def treat_data(ids) -> tuple:
    """
    Join all the data from the users as a sequential dataset.
    ├── user1
    │   ├── train_loader 1
    │   └── test_loader 1
    ├── user2
    │   ├── train_loader 2
    │   └── test_loader 2

    [train_loader 1, train_loader 2, ...] [test_loader 1, test_loader 2, ...]

    Returns:
    - train_dataset: List of train datasets.
    - test_dataset: List of test datasets.
    """
    train_dataset = []
    test_dataset = []
    # Join all the data from the users as a sequential dataset
    for id in ids:
        train_loader, test_loader = ids[id]
        train_dataset.append(train_loader)
        test_dataset.append(test_loader)

    return train_dataset, test_dataset


def train(model, train_loader, params, device):
    """ Training function adapted to the centralized scenario."""
    # Setup a loss function
    loss_fn = torch.nn.MSELoss()
    # Setup an optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.lr)
    # Bulding a training loop in PyTorch
    for epoch in range(params.epochs):
        for train_ in train_loader:
            model.train()
            for batch in train_:
                # Get the data and labels
                X_train, y_train = batch
                # Move the data and labels to GPU
                X_train = X_train.unsqueeze(2).to(device)
                y_train = y_train.to(device)
                y_pred = model(X_train)
                loss = loss_fn(y_pred.reshape(y_pred.shape[0], -1), y_train)
                # Optimizer zero grad
                optimizer.zero_grad()
                # Perform backpropagation on the loss with respect to the parameters of the model
                loss.backward()
                # Step the optimizer
                optimizer.step()


def centralized(ids, params, plotting=False):
    # resultados_excel = pd.DataFrame(columns=['User', 'mse', 'mae', 'mape', 'smape'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    # organise dataset
    train_loader, _ = treat_data(ids)

    # Train and test several times to reduce the randomness
    accumulated_errors = []
    metricas_totales = []

    net = instantiate(params.model).to(device)

    inicio = time.time()
    train(net, train_loader, params, device)
    fin = time.time()
    for id in ids:
        _, test_loader = ids[id]
        loss, metrics = test(net, test_loader, user=id)
        accumulated_errors.append(loss)
        metricas_totales.append(metrics)
        # print(f'User: {id} | Test loss: {loss:.4f}')
        diccionario = metrics.copy()
        diccionario['User'] = id
        diccionario['mse'] = loss
        # resultados_excel.loc[len(resultados_excel)] = diccionario

    # Plot results
    # plotear_res_final(modelo, "centralized", plotear=plotting)

    # Save the model
    model_name = params.model['_target_'].split('.')[-2]
    torch.save(net.state_dict(), f"configs/models/{model_name}_centralized.pth")

    # Calculate mean metrics
    mean_metricas = mean_metrics(metricas_totales) # calculate metrics mean values

    # Print results
    # print(f'Mean loss (MSE): {np.mean(accumulated_errors):.4f}')
    # print(f'Standard deviation: {np.std(accumulated_errors):.4f}')
    # print("Media de las métricas:")
    # for metrica, valor_medio in mean_metricas.items():
    #     print(f"{metrica}: {valor_medio}")

    # diccionario = mean_metricas.copy()
    # diccionario['User'] = 'Mean'
    # diccionario['mse'] = np.mean(accumulated_errors)
    # resultados_excel.loc[len(resultados_excel)] = diccionario
    # resultados_excel.to_excel("results/centralized.xlsx", index=False, sheet_name='Centralized')

    execution_time = fin - inicio
    # print(f"Tiempo de ejecución: {execution_time:.4f} segundos")
    return {'MSE': np.mean(accumulated_errors), 'std': np.std(accumulated_errors),'metrics': mean_metricas, 'time': execution_time}