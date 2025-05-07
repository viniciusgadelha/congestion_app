############################# CODE TO MANAGE DATASET #############################
"""
Utils to manage the dataset and prepare data as input for the TCN model.

Functions:
client_df(df: pd.DataFrame, freq='30m') -> pd.DataFrame: Function to create a dataframe with the clients as columns and the dates as index. The values are the KWh consumed.
escalar(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple: Function to scale the dataset. The StandardScaler from sci-kit learn is used for this purpose.
window(arr: np.ndarray, out_size= 24, inp_size=192, step=24) -> tuple[TensorDataset, TensorDataset, TensorDataset]: Organize the input dataset for the models.
treat_missing_values(df: pd.DataFrame, elim_percent=False) -> pd.DataFrame: Function to find missing values. If the ratio of missing values is higher than 10%, the user will be deleted.
miss_out_val(df: pd.DataFrame, r=1.5, treat_missing_values=False) -> pd.DataFrame: Function to extract outliers and missing values from the dataset.
load_data() -> dict: Function to load the data and feed the models. It will try to read the file *client_dataframe.csv*. If not, it will create the file using the *london_energy.csv* dataset file.
"""
# Libraries
from typing import List, Tuple
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import torch
import os
import glob


def read_single_csv():
    folder_path = 'dataset'
    # Use glob to find all .csv files in the folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # Check the number of .csv files found
    if len(csv_files) == 0:
        raise FileNotFoundError("No .csv files found in the folder.")
    elif len(csv_files) > 1:
        raise FileExistsError("Multiple .csv files found in the folder.")

    # Read the single .csv file
    df = pd.read_csv(csv_files[0], header=0, parse_dates=['time'], index_col=['time'])
    return df


def client_df(df: pd.DataFrame, freq='30m'):
    """ Function to create a dataframe with the clients as columns and the dates as index. The values are the KWh consumed.
    The frequency of the dataset can be changed (30 min is the default value). """
    date_rng = pd.date_range(start=df.index[0], end=df.index[-1],
                             freq=freq)  # Create a date range to detect missing values
    list_clients = df['id'].unique()
    list_clients = np.random.choice(list_clients, size=20, replace=False)
    client_dataframe = pd.DataFrame(index=date_rng)
    client_dataframe.index.name = 'time'
    for user in tqdm(list_clients):
        df_p = df[df['id'] == user]
        # Eliminate id column
        df_p = df_p.drop(['id'], axis=1)
        # Change the name of the variable column
        df_p = df_p.rename(columns={'energy': user})
        # eliminate duplicates
        df_p = df_p[~df_p.index.duplicated(keep='first')]
        # Add column to the dataframe
        client_dataframe = pd.merge(client_dataframe, df_p, right_index=True, left_index=True, how='outer')
    client_dataframe = client_dataframe.dropna(how='all')
    return client_dataframe


def escalar(train: np.ndarray, test: np.ndarray):
    """ Function to scale the dataset. The StandardScaler from sci-kit learn is used for this purpose.
    The Yeo-Johnson transformation is used to normalize the variables
    Additionally, the dataset is standardized.
    Parameters
    ----------
    train : np.ndarray
        Numpy array with the training data.
    val : np.ndarray
        Numpy array with the validation data.
    test : np.ndarray
        Numpy array with the test data.
        ..
    Output
    ------
    scaled_train : np.ndarray
        Numpy array with the scaled training data.
    scaled_val : np.ndarray
        Numpy array with the scaled validation data.
    scaled_test : np.ndarray
        Numpy array with the scaled test data.
    """
    # Estandarizar valores con el PowerTransformer de scikit-learn
    # pt = PowerTransformer(method='yeo-johnson', standardize=True)
    pt = StandardScaler()
    scaled_train = pt.fit_transform(train)
    scaled_test = pt.transform(test)
    return scaled_train, scaled_test, pt


def window(arr: np.ndarray, out_size=24, inp_size=192, step=24):
    """Organize the input dataset for the models. By default, the input size is 8 days and the output size is 24h. Data
    is split into train, validation and test and is scaled using the StandardScaler from sci-kit learn. The output of
    the function will be four TensorDataset: the train, validation and test data for energy values and the days
    of the weeks that have been used and the one to be predicted.

    Parameters
    ----------
    out_size : int
        Number of output values (by default are 24 hours prediction).
    step : int
        The step between train values. Should be the same as the output size.
    inp_size : int
        Number of values for the input data (by default is 192 which are one week and one day.
        ..

    Output
    ------
    train_dataset : TensorDataset
        Dataset with the training data.
    val_dataset : TensorDataset
        Dataset with the validation data.
    test_dataset : TensorDataset
        Dataset with the test data.
    """
    id_arr = arr.name
    device = "cuda" if torch.cuda.is_available() else "cpu"
    days = arr.index.to_numpy()
    days = (days.astype('datetime64[D]').view('int64') - 4) % 7  # Days from 0 to 6
    arr = arr.values
    # Create empty file where others will be appended
    X, y, d = np.empty((0, inp_size)), np.empty((0, out_size)), np.empty((0, 1))
    # Hecho para las salidas de un solo valor
    for input_v in range(0, len(arr), step):
        try:
            y = np.append(y, arr[input_v + inp_size:input_v + inp_size + step].reshape(1, -1), axis=0)
            X = np.append(X, arr[input_v:input_v + inp_size].reshape(1, -1), axis=0)
            d = np.append(d, days[input_v + inp_size])
        except:
            pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    d_train, d_test = train_test_split(d, test_size=0.3, shuffle=False)

    folder_path = '.cache'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Estandarizar valores de entrada con el Scaler de scikit-learn
    X_train, X_test, pt_X = escalar(X_train, X_test)
    file_name = os.path.join(folder_path, 'pt_X'+id_arr+'.pkl')
    joblib.dump(pt_X, file_name)
    # Estandarizar valores objetivos con el Scaler de scikit-learn
    y_train, y_test, pt_y = escalar(y_train, y_test)
    file_name = os.path.join(folder_path, 'pt_y'+id_arr+'.pkl')
    joblib.dump(pt_y, file_name)
    # Estandarizar valores de entrada con el Scaler de scikit-learn
    d_train, d_test, pt_d = escalar(d_train.reshape(-1, 1), d_test.reshape(-1, 1))

    X_train = np.concatenate((X_train, d_train), axis=1)
    X_test = np.concatenate((X_test, d_test), axis=1)

    # Convertir las matrices NumPy a tensores de PyTorch
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    # Crear conjuntos de datos de PyTorch para entrenamiento, validación y prueba
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    return train_dataset, test_dataset  # , train_dataset_d, val_dataset_d, test_dataset_d


def treat_missing_values(df: pd.DataFrame, elim_percent=False):
    """ Function to find missing values. If the ratio of missing values is higher than 10%, the user will be deleted.
    If the ratio is lower than 10%, the missing values will be interpolated. Missing values at the beginning or the
    end of the dataset can not be interpolated. Therefore, they will be deleted. That will cause that the dataset will
    be smaller."""
    if elim_percent:
        # Show missing values
        miss_val = df.isnull().sum()
        # Know the percentage of clients with more than 10% of missing values
        miss_val = miss_val * 100 / len(df)
        miss_val_limit = miss_val[miss_val > 10]  # For users with more than 10% of missing values --> delete
        print(
            f"For limit = 10% of the dataset, users that will de deleted are the {miss_val_limit.nunique() * 100 / len(df.columns):.2f}% of the dataset")
        # Delete users with more than 10% of missing values
        df = df.drop(columns=miss_val_limit.index)
    return df.dropna()


def miss_out_val(df: pd.DataFrame, r=1.5, treat_missing_values=False):
    """ Function to extract outliers and missing values from the dataset. The outliers are detected using the IQR
    method. It has de possibility of eliminate missing values.
    Parameters
    ----------
    r : float
        Values that multiply the IQR and will eliminate further values.
    treat_missing_values : bool
        If True, the missing values will be treated.
        ..

    Output
    ------
    df : pd.DataFrame
        Dataset without outliers and missing values.
    """
    print("\n\x1b[31mExtraer outliers del dataset:\x1b[0m")
    for columna in tqdm(df.columns):
        q1 = df[columna].quantile(0.25)
        q3 = df[columna].quantile(0.75)
        iqr = q3 - q1
        umbral_inferior = q1 - r * iqr
        umbral_superior = q3 + r * iqr
        outliers = df[(df[columna] < umbral_inferior) | (df[columna] > umbral_superior)]
        indices_a_eliminar = outliers.index.tolist()
        df.loc[df.index.isin(indices_a_eliminar), columna] = None
    if treat_missing_values:
        # Treat missing values
        df = treat_missing_values(df)
    return df


def load_data():
    """Function to load the data and feed the models. It will try to read the file *client_dataframe.csv*. If not, it will
    create the file using the *london_energy.csv* dataset file. After that, the dataset missing values will be treated
    removing those that are can't be interpolated because of not having data surrounding it and interpolate those that
    are feasible. Finally, data will be organized using torch.utils.data.DataLoader for convenience.

    Output
    ------
    ids : dict
        Dictionary with different mac´s id and their train, validation and test DataLoaders to feed the models.
    """
    # Read the dataframe
    try:
        client_dataframe = read_single_csv()
    except (FileNotFoundError, FileExistsError) as e:
        print(e)


    # Encontrar missing values y outliers (para outliers r=1.5)
    # client_dataframe = miss_out_val(client_dataframe, r=10)

    ids = dict()
    # loop to separate the clients
    for id in client_dataframe.columns:
        # Select id
        client = client_dataframe[id]
        # client = client[~client.index.duplicated(keep='first')]
        first_idx = client.first_valid_index()
        last_idx = client.last_valid_index()
        client = client.loc[first_idx:last_idx]
        client = client.interpolate(method='linear', limit_direction='forward', axis=0)
        client = client.resample('h', label='right', closed='right').sum()
        first_midnight = client[client.index.time == pd.Timestamp('00:00:00').time()].index[0]
        client = client[first_midnight:]
        try:
            train_dataset, test_dataset = window(client)
        except:
            continue

        # Crear cargadores de datos (DataLoaders) para cada conjunto
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # Append to dictionary
        ids[id] = (train_loader, test_loader)
    return ids