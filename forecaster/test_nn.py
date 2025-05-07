from forecaster.short_term_load_forecasting.utils.manage_dataset import escalar
from forecaster.short_term_load_forecasting.models.TCN import TCN
import argparse
import numpy as np
import os
import glob
import re
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
import warnings
import openpyxl

# Suprimir warnings de tipo DeprecationWarning
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=FutureWarning)

def plotear_res_final(name, y_test, y_pred):
    y_pred = y_pred.reshape(y_pred.shape[0], -1)

    scaler = joblib.load('.cache/pt_y' + name + '.pkl')
    y_test = scaler.inverse_transform(y_test)
    y_pred = scaler.inverse_transform(y_pred)

    test_data = y_test[-1,:].flatten()
    test_results = y_pred[-1,:].flatten()

    plt.plot(test_data, label="Real")
    plt.plot(test_results, label="Pred", linestyle='dashed', color='red')
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Energy consumption")
    plt.show()


def run_forecast(framework):
    file_path = 'dataset/test/ct_prueba.csv'
    df = pd.read_csv(file_path, header=0, parse_dates=['time'], index_col=['time'])

    ids = dict()
    # loop to separate the clients
    for id in df.columns:
        if framework == 'federated':
            files = glob.glob(".cache/model_round_fedavg*")
            model_path = max(files, key=lambda x: int(re.search(r'\d+', x).group()))
        else:
            if framework == 'centralized':
                model = 'TCN_centralized'
            elif framework == 'isolated':
                model = f'TCN_{id}_isolated'
            model_path = f'configs/models/{model}.pth'  # Modelo guardado en formato .pth de PyTorch

        model = TCN()
        model.load_state_dict(torch.load(model_path))
        # Select id
        client = df[id]
        # client = client[~client.index.duplicated(keep='first')]
        first_idx = client.first_valid_index()
        last_idx = client.last_valid_index()
        client = client.loc[first_idx:last_idx]
        client = client.interpolate(method='linear', limit_direction='forward', axis=0)
        client = client.resample('h', label='right', closed='right').sum()
        first_midnight = client[client.index.time == pd.Timestamp('00:00:00').time()].index[0]
        client = client[first_midnight:]
        try:
            id_arr = client.name
            device = "cuda" if torch.cuda.is_available() else "cpu"
            days = client.index.to_numpy()
            days = (days.astype('datetime64[D]').view('int64') - 4) % 7  # Days from 0 to 6
            arr = client.values
            # Create empty file where others will be appended
            X, y, d = np.empty((0, 192)), np.empty((0, 24)), np.empty((0, 1))
            # Hecho para las salidas de un solo valor
            for input_v in range(0, len(arr), 24):
                try:
                    y = np.append(y, arr[input_v + 192:input_v + 192 + 24].reshape(1, -1), axis=0)
                    X = np.append(X, arr[input_v:input_v + 192].reshape(1, -1), axis=0)
                    d = np.append(d, days[input_v + 192])
                except:
                    pass

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, shuffle=False)
            d_train, d_test = train_test_split(d, test_size=1, shuffle=False)

            folder_path = '.cache'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Estandarizar valores de entrada con el Scaler de scikit-learn
            X_train, X_test, pt_X = escalar(X_train, X_test)
            # Estandarizar valores objetivos con el Scaler de scikit-learn
            y_train, y_test, pt_y = escalar(y_train, y_test)
            # Estandarizar valores de entrada con el Scaler de scikit-learn
            d_train, d_test, pt_d = escalar(d_train.reshape(-1, 1), d_test.reshape(-1, 1))

            X_test = np.concatenate((X_test, d_test), axis=1)

            X_test = torch.Tensor(X_test).to(device)
            y_test = torch.Tensor(y_test).to(device)
            test_dataset = TensorDataset(X_test, y_test)
            model.eval()  # Ponemos el modelo en modo de evaluación
            batch_size = 64
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():  # Desactivamos el cálculo del gradiente
                for batch in test_loader:
                    X, y = batch
                    X = X.unsqueeze(2)
                    predicciones = model(X)
            predicciones = predicciones.reshape(predicciones.shape[0], -1)
            scaler = joblib.load('.cache/pt_y' + id + '.pkl')
            predicciones = scaler.inverse_transform(predicciones)
        except:
            predicciones = np.zeros((1, 24))
        ids[id] = predicciones

    ids = {key: value[0].tolist() for key, value in ids.items()}
    df = pd.DataFrame(ids)
    df.to_excel(f'results/predicciones_{framework}.xlsx', index=False)
    # plotear_res_final(ids, y, predicciones)
    return ids  # Convertimos las predicciones a un array de numpy



if __name__ == '__main__':
    # Crear el parser
    # parser = argparse.ArgumentParser(description="Parameters of frameworks to test")
    # parser.add_argument("framework", type=str, help="Framework to test")
    # args = parser.parse_args()
    # run_forecast(args.framework)
    run_forecast("centralized")