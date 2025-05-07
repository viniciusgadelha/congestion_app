#########################PREDICTION WITH TCN#########################
# Librerías
import torch
import argparse
import pandas as pd
import numpy as np
# from openpyxl import load_workbook
from .utils.manage_dataset import load_data, window, miss_out_val, escalar
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from TCN import TCN, LSTMMIMO
import time
from .utils.utils import give_metrics, mean_metrics, plotear_res_final
from .client import train, test
from hydra.utils import instantiate

# Use graphs edition
# from graficas import *
# plotear = False

"""
Isolated scenario where users train their own model with their own data. Communication between users does not exist.
Neither with the central server.
The input parameters of this files are the number of epochs and the model used. Number of epochs is 300 by default,
but can be any other integer. On the other hand, the model is TCN by default, but can be a LSTM model. To run the model, use
the following command:
    python isolated.py --epochs 30 --model TCN
    python isolated.py --epochs 30 --model LSTM

Functions used in this code:
train(model, train_loader, epochs, device): Training function.
test(model, test_loader, device): Test function.
main(model, epochs): Main function to run the isolated scenario.
"""

def isolated(ids, params, plotting=False):
    # resultados_excel = pd.DataFrame(columns=['User', 'mse', 'mae', 'mape', 'smape'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    lossess = []
    metricas_totales = []
    tiempos = []
    for id in ids:
        net = instantiate(params.model).to(device)
        train_loader, test_loader = ids[id]
        inicio = time.time()
        train(net, train_loader, params.epochs, params.lr)
        fin = time.time()
        tiempos.append(fin - inicio)
        loss, metrics = test(net, test_loader, user=id)
        lossess.append(loss)
        metricas_totales.append(metrics)
        # Check if the user is the selected one
        # if id == "MAC002393":
        #     plotear_res_final(net, "isolated", plotear=plotear, usuarios=[id])
        # if id == "MAC000159":
        #     plotear_res_final(net, "isolated", plotear=plotear, usuarios=[id])
        # if id == "MAC003032":
        #     plotear_res_final(net, "isolated", plotear=plotear, usuarios=[id])
        # if id == "MAC004429":
        #     plotear_res_final(net, "isolated", plotear=plotear, usuarios=[id])
        # print(f'User: {id} | Test loss: {loss:.4f}')
        # diccionario = metrics.copy()
        # diccionario['User'] = id
        # diccionario['mse'] = loss
        # resultados_excel.loc[len(resultados_excel)] = diccionario
        model_name = params.model['_target_'].split('.')[-2]
        torch.save(net.state_dict(), f"configs/models/{model_name}_{id}_isolated.pth")

    mean_metricas = mean_metrics(metricas_totales)  # calculate metrics mean values

    # Mean metrics
    # print(f'Mean loss (MSE): {np.mean(lossess):.4f}')
    # print(f'Standard deviation: {np.std(lossess):.4f}')
    # print("Media de las métricas:")
    # for metrica, valor_medio in mean_metricas.items():
    #     print(f"{metrica}: {valor_medio}")

    # diccionario = mean_metricas.copy()
    # diccionario['User'] = 'Mean'
    # diccionario['mse'] = np.mean(lossess)
    # resultados_excel.loc[len(resultados_excel)] = diccionario
    # resultados_excel.to_excel("final_results/isolated.xlsx", index=False, sheet_name='Isolated')

    # print(f"Tiempo de ejecución: {max(tiempos):.4f} segundos") # cogemos el qué mas tiempo estuvo realizando el ejc
    return {'MSE': np.mean(lossess), 'std': np.std(lossess),'metrics': mean_metricas, 'time': max(tiempos)}


