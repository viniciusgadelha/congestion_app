"""
Functions:
mean_absolute_percentage_error(y_true, y_pred) -> float: Calculate the mean absolute percentage error (MAPE) for a batch.
s_mean_absolute_percentage_error(y_true, y_pred) -> float: Calculate the symmetric mean absolute percentage error (sMAPE) for a batch.
give_metrics(test_pred, y_test) -> tuple: Function that gives mae, mape and smape metrics.
mean_metrics(metrics: list) -> dict: Mean of the metrics.
plotear_res_final(model, plotear=False, case_study): Plot the real and predicted values of the energy consumption and save comparison result.
"""
import os
from permetrics.regression import RegressionMetric
import numpy as np
import torch
from .manage_dataset import load_data
import matplotlib.pyplot as plt
import pickle
import joblib

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the mean absolute percentage error (MAPE) for a batch.

    Parameters:
    - y_true: Tensor of real values.
    - y_pred: Tensor of predicted values.

    Returns:
    - MAPE: Mean Absolute Percentage Error.
    """

    y_true = y_true.cpu().numpy()
    y_pred = y_pred.reshape(y_pred.shape[0], -1).cpu().numpy()

    # Asegurarse de que las dimensiones sean consistentes
    if y_true.shape != y_pred.shape:
        raise ValueError("Las dimensiones de y_true y y_pred deben ser las mismas.")

    # Evitar divisiones por cero al agregar una pequeña cantidad
    epsilon = 1e-10

    # Calcular el MAPE para cada batch y promediar
    mape_per_batch = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon)) * 100, axis=1)
    mape = np.mean(mape_per_batch)

    return mape

def s_mean_absolute_percentage_error(y_true, y_pred):
    """
        Calculate the symmetric mean absolute percentage error (sMAPE) for a batch.

        Parameters:
        - y_true: Tensor of real values.
        - y_pred: Tensor of predicted values.

        Returns:
        - sMAPE: symmetric mean absolute percentage error
        """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.reshape(y_pred.shape[0], -1).cpu().numpy()

    # Asegurarse de que las dimensiones sean consistentes
    if y_true.shape != y_pred.shape:
        raise ValueError("Las dimensiones de y_true y y_pred deben ser las mismas.")

    # Evitar divisiones por cero al agregar una pequeña cantidad
    epsilon = 1e-10

    # Calcular el sMAPE para cada batch y promediar
    num = np.abs(y_true - y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape_per_batch = np.mean(num / (denom + epsilon) * 100, axis=1)
    return np.mean(smape_per_batch)


def give_metrics(test_pred, y_test):
    """
    Function that gives mae, mape and smape metrics.

    Parameters
    ----------
    test_pred : np.array
        Predictions of the model.
    y_test : np.array
        Real values.

    Returns
    -------
    mae : float
        Mean absolute error.
    mape : float
        Mean absolute percentage error.
    smape : float
        Symmetric mean absolute percentage error.
    """
    evaluator = RegressionMetric(test_pred, y_test)
    mae = np.mean(evaluator.MAE(multi_output="raw_values"))
    mape = np.mean(evaluator.MAPE(multi_output="raw_values"))
    smape = np.mean(evaluator.SMAPE(multi_output="raw_values"))
    return mae, mape, smape

def mean_metrics(metrics: list) -> dict:
    """
    Mean of the metrics

    Parameters
    ----------
    metrics : list
        List of metrics.

    Returns
    -------
    mean_metricas : dict
        Mean of the metrics.
    """
    mean_metricas = {}
    for metric in metrics[0]:
            mean_metricas[metric] = np.mean([m[metric] for m in metrics])
    return mean_metricas

def plotear_res_final(model: torch.nn.Module, case_study:str, plotear: bool=False, usuarios="todos") -> None:
    """
    Plot the real and predicted values of the energy consumption. It also saves the results in a folder called
    final_results as a pickle file.
    :param model: torch.nn.Module
    :param plotear: bool
    :return:
    """
    if usuarios == "todos":
        lista_usuarios = ["MAC002393", "MAC000159", "MAC003032", "MAC004429"]
    else:
        lista_usuarios = usuarios

    for usuario in lista_usuarios:

        id = load_data()
        _, test_loader = id[usuario]
        X_test, y_test = next(iter(test_loader))
        if case_study != 'ifca':
            X_test = X_test.unsqueeze(2)
        # Make a prediction
        with torch.inference_mode():  # disable useless things of the model and only shows the predictions
            y_pred = model(X_test)
        y_pred = y_pred.reshape(y_pred.shape[0], -1)

        scaler = joblib.load('.cache/pt_y' + usuario + '.pkl')
        y_test = scaler.inverse_transform(y_test)
        y_pred = scaler.inverse_transform(y_pred)

        list_days = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5]
        if case_study == 'ifca':
            days = X_test[4:7,-1].flatten().numpy().round(2)
        else:
            days = X_test[4:7,-1,:].flatten().numpy().round(2)
        indices = []
        for day in days:
            try:
                indice = list_days.index(day)
                indices.append(indice)
            except ValueError:
                indices.append(None)

        np.save(".cache/days_"+usuario+".npy", indices)

        test_data = y_test[4:7,:].flatten()
        test_results = y_pred[4:7,:].flatten()
        if plotear:
            plt.plot(test_data, label="Real")
            plt.plot(test_results, label="Pred")
            plt.legend()
            plt.xlabel("Time")
            plt.ylabel("Energy consumption")
            plt.show()

        # create a folder called final_results if it doesn't exist
        if not os.path.exists("final_results"):
            os.makedirs("final_results")
        path_name = "final_results/prediction_"+case_study+"_"+usuario+".pkl"
        print('Saving results in:', path_name)
        # save y_pred[4,:] and y_test[4,:] as a pickle file
        with open(path_name, "wb") as f:
            pickle.dump(test_results, f)
        with open("final_results/real_"+usuario+".pkl", "wb") as f:
            pickle.dump(test_data, f)