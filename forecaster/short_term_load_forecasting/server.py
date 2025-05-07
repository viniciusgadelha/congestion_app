# Libraries
import torch
import os
import glob
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Callable, Optional, Dict, Union
from flwr.common import Metrics
import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from .models.TCN import TCN
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
# from logging import WARNING, INFO
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log
from functools import reduce
import random
import threading
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

number_clusters = 4

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

net = TCN()

class FedAvgSaveModel(fl.server.strategy.FedAvg):
    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            folder_path = '.cache'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_name = f'model_round_fedavg{server_round}.pth'
            file_path = os.path.join(folder_path, file_name)

            # Save the model
            torch.save(net.state_dict(), file_path)

        return aggregated_parameters, aggregated_metrics


class AdjustedClientManager(ClientManager):
    """Provides a pool of available clients."""
    """
    Create the ClientManager to dont shuffle and select all the clients in each round.
    La única diferencia es que sample no elige usuarios y los baraja sino que siempre son los mismos.
    Esto disminuye la privacidad seguramente pero mejora la exactitud de los modelos.
    """

    def __init__(self) -> None:
        self.clients: Dict[str, ClientProxy] = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(self, num_clients: int, min_num_clients: Optional[int] = None, **kwargs) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)  # no hace un sample sino que coge siempre los mismos
        return [self.clients[cid] for cid in available_cids]

def h_clustering_method(matriz_distancias, threshold=0.325):
    """
    Hierarchical cluster de ids. El marco esta preparado para dos clusters por el método de agregación y selección de clientes
    de la estrategia. Sin embargo, en esta parte del código se pueden añadir más de un cluster.

    Parameters:
    ----------
    matriz_distancias: np.array
        Matrix with euclidian distances between each point. The matrix must be symetrical.
    num_clusters: int=2
        Number of clusters created.

    Returns:
    ----------
    indices_mas_pequenos_por_fila:
        List with the cluster that each id belongs to.
    """
    if len(matriz_distancias) < threshold:
        raise ValueError("Matrix lenght is smaller than number of clusters")
    threshold = threshold
    dist_array = squareform(matriz_distancias)
    # Realizar el clustering jerárquico
    Z = linkage(dist_array, 'single')
    # clusters = fcluster(Z, threshold, criterion='distance')
    clusters = fcluster(Z, number_clusters, criterion='maxclust')
    # También puedes guardar la asignación de clusters en una lista si lo prefieres
    indices_mas_pequenos_por_fila = clusters.tolist()
    # Visualizar el dendrograma
    # plt.figure(figsize=(10, 7))
    # dendrogram(Z)
    # plt.title('Dendrograma')
    # plt.xlabel('Índice de la muestra')
    # plt.ylabel('Distancia')
    # plt.show()

    print("Índices de los valores más pequeños por fila:", indices_mas_pequenos_por_fila)

    return indices_mas_pequenos_por_fila


def clustering_method(matriz_distancias, num_clusters=number_clusters):
    """
    Cluster de ids. El marco esta preparado para dos clusters por el método de agregación y selección de clientes
    de la estrategia. Sin embargo, en esta parte del código se pueden añadir más de un cluster.

    Parameters:
    ----------
    matriz_distancias: np.array
        Matrix with euclidian distances between each point. The matrix must be symetrical.
    num_clusters: int=2
        Number of clusters created.

    Returns:
    ----------
    indices_mas_pequenos_por_fila:
        List with the cluster that each id belongs to.
    """
    if len(matriz_distancias) < num_clusters:
        raise ValueError("Matrix lenght is smaller than number of clusters")
    n = num_clusters - 1
    indices_superiores = np.triu_indices(len(matriz_distancias), k=1)
    # Obtener los valores de la mitad superior de la matriz
    valores_superiores = matriz_distancias[indices_superiores]
    # Encontrar los índices de los n valores más grandes en la mitad superior de la matriz
    indices_max_valores = np.argpartition(valores_superiores, -n)[-n:]
    # Obtener los valores de la matriz de distancias correspondientes a los índices obtenidos
    max_valores = valores_superiores[indices_max_valores]
    # Obtener los índices originales de la matriz de distancias correspondientes a los índices obtenidos
    max_indices_originales = (indices_superiores[0][indices_max_valores], indices_superiores[1][indices_max_valores])
    # Crear un conjunto de índices para eliminar duplicados
    conjunto_indices = set(max_indices_originales[0]).union(set(max_indices_originales[1]))
    # Convertir el conjunto en una lista de índices finales
    indices_finales = list(conjunto_indices)

    columnas_seleccionadas = matriz_distancias[:, indices_finales]

    indices_mas_pequenos_por_fila = []  # A que cluster pertenecen

    # Iterar sobre cada fila
    for i in range(columnas_seleccionadas.shape[0]):
        # Obtener los valores en la fila actual
        fila_actual = columnas_seleccionadas[i]
        # Encontrar el índice del valor más pequeño en la fila actual
        indice_mas_pequeno = np.argmin(fila_actual)
        # Agregar el índice del valor más pequeño a la lista
        indices_mas_pequenos_por_fila.append(indice_mas_pequeno)

    # Revisar que el número de clusters sea el correcto
    for i, indice in enumerate(indices_mas_pequenos_por_fila):
        if indice >= num_clusters:
            indices_mas_pequenos_por_fila[i] = num_clusters - 1

    print("Índices de los valores más pequeños por fila:", indices_mas_pequenos_por_fila)

    return indices_mas_pequenos_por_fila


def distancia_euclidiana_media(resultados):
    """
    distancia_euclidiana_media es una función que crea una matriz simétrica con las distancias euclidianas entre los usuarios. Tiene como input una lista cuya longitud
    es el número de usuarios incluidos en el marco. Cada item de la lista tiene una lista de dos items. El primero aloja los resultados obtenidos de los parámetros entrenados
    de cada usuario. El segundo tiene la longitud del dataset con el que ha entrenado cada usuario para ponderar la agregación. En esta función, se asume que las dimensiones
    de los modelos es la misma para todos los usuarios. A continuación, se calcula la distancia euclidiana entre cada pareja de usuarios como la media entre las distancias
    euclidianas de cada matriz perteneciente a cada capa del modelo.

    Parameters:
    ----------
    resultados: List
        List of the results returned after each round.

    Returns:
    ----------
    indices_mas_pequenos_por_fila:
        List with the cluster that each id belongs to.
    """
    # Inicializar una lista para almacenar las distancias euclidianas de cada matriz
    distancias_matriz = []
    n_usuarios = len(resultados)
    usuar_ = resultados[0]
    n_matrices = len(usuar_[0])

    diction_usuarios = {}
    for usuario in range(n_usuarios):
        usuario_ = resultados[usuario]
        matriz_usuario_uno = usuario_[0]
        diction_usuarios[usuario] = matriz_usuario_uno

    # Crear una matriz para almacenar las distancias entre los usuarios
    matriz_distancias = np.zeros((n_usuarios, n_usuarios))

    # Calcular las distancias euclidianas entre cada par de usuarios
    for i, (usuario1, matrices_usuario1) in enumerate(diction_usuarios.items()):
        for j, (usuario2, matrices_usuario2) in enumerate(diction_usuarios.items()):
            # Calcular la distancia euclidiana entre los conjuntos de matrices de los usuarios
            distancia_usuario = 0
            for matriz1, matriz2 in zip(matrices_usuario1, matrices_usuario2):
                distancia_usuario += np.linalg.norm(matriz1 - matriz2)
            # Calcular la distancia promedio entre las matrices de los usuarios
            distancia_promedio = distancia_usuario / len(matrices_usuario1)
            # Asignar la distancia promedio a la matriz de distancias
            matriz_distancias[i, j] = distancia_promedio

    # Imprimir la matriz de distancias
    print("Matriz de distancias:")
    print(matriz_distancias)
    indices_mas_pequenos_por_fila = clustering_method(matriz_distancias)

    return indices_mas_pequenos_por_fila

def cluster_method_Sara(resultados):
    distancias_matriz = []
    n_usuarios = len(resultados)
    usuar_ = resultados[0] # Asumiendo que todos los usuarios tienen las mismas redes neuronales
    n_matrices = len(usuar_[0])

    diction_usuarios = {}
    for usuario in range(n_usuarios):
        usuario_ = resultados[usuario]
        matriz_usuario_uno = usuario_[0]
        first_mean = np.array([])
        second_mean = np.array([])
        # media de los pesos de la primera parte de la red
        for layer in range(len(resultados)//2):
          first_mean = np.append(first_mean, np.mean(matriz_usuario_uno[layer]))
        # media de los pesos de la primera parte de la red
        for layer in range(len(resultados)-len(resultados)//2):
          second_mean = np.append(second_mean, np.mean(matriz_usuario_uno[len(resultados)//2+layer]))
        diction_usuarios[usuario] = (np.mean(first_mean), np.mean(second_mean))
        # plt.plot(np.mean(first_mean), np.mean(second_mean), 'bo')
        # plt.show()
    valores_array = np.array(list(diction_usuarios.values()))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=number_clusters)
    kmeans.fit(valores_array)
    clusters = kmeans.labels_
    clusters = clusters.tolist()
    print(clusters)
    y_kmeans = kmeans.predict(valores_array)
    plt.scatter(valores_array[:, 0], valores_array[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.show()

    return clusters

def aggregate_fedmax(results: List[Tuple[NDArrays, int]], last_clusters, server_round, limit) -> NDArrays:
    """
    Function to create the aggregation of the clients. It is doing the FedAvg aggregation taking the results of the
    clients. The difference between FedAvg and this on is that here, the aggregation is done after the euclidian distances measure.
    The aggregation itself is the same,
    Parameters:
        ----------
        results: List
            Matrix with euclidian distances between each point. The matrix must be symmetrical.
        last_clusters: List
            Last cluster list that was measured.
        server_round: int
            Number round that the model is running.
        limit: init
            Server round limit to stop doing the cluster and training the global model to start training the clusters.

        Returns:
        ----------
        weights_prime:
            Weights of the models after aggregation.
        clusters: List
            List with the last clusters measured which each user belongs to
    """
    clusters = last_clusters
    if server_round <= limit:
        print(10 * '--')
        clusters = distancia_euclidiana_media(results)
        # clusters = cluster_method_Sara(results)
        print(10 * '--')

    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime, clusters


# Strategy
class FedMax(Strategy):
    """Adaptation of FedMax strategy.

    Implementation based on doi:10.1109/CLOUD49709.2020.00064

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
            self,
            *,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            inplace: bool = False,
    ) -> None:
        super().__init__()

        if (
                min_fit_clients > min_available_clients
                or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.inplace = inplace
        self.clusters = [None] * self.min_fit_clients  # lista de variables del cluster
        self.round_cluster = 5  # Límite de ronda para el entrenamiento global y del cluster
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedMax(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    # TODO: cambiar los valores iniciales del modelo
    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        return None

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Resetear parameters para cada cluster
        if server_round > self.round_cluster and server_round % self.round_cluster == 1:
            list_of_files = [fname for fname in glob.glob("saved_models/model_round_*")]
            latest_round_file = max(list_of_files, key=os.path.getctime)
            print("Loading pre-trained model from: ", latest_round_file)
            state_dict = torch.load(latest_round_file)
            net.load_state_dict(state_dict)
            state_dict_ndarrays = [v.cpu().numpy() for v in net.state_dict().values()]
            parameters = fl.common.ndarrays_to_parameters(state_dict_ndarrays)

        lista_fit_ins = []
        for conf_ in self.clusters:
            config_ = {"cluster": conf_}
            fit_ins = FitIns(parameters, config_)
            lista_fit_ins.append(fit_ins)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if server_round > self.round_cluster:  # cambiar esto por cluster
            print(10* '---')
            print('Num clusters: ', len(set(self.clusters)))
            print(10 * '---')
            # Crear clusters
            grupos = {}
            for cluster, cliente in zip(self.clusters, clients):
                # Si el cluster aún no está en el diccionario, creamos una nueva lista para él
                if cluster not in grupos:
                    grupos[cluster] = []
                # Añadimos el cliente a la lista correspondiente al valor en el diccionario
                grupos[cluster].append(cliente)
            # if (server_round > self.round_cluster) and (server_round <= self.round_cluster * 2):
            #     clients = list(grupos.values())[0]  # selección del primer grupo
            # else:
            #     clients = list(grupos.values())[1]
            # Calcular el índice del cluster al que pertenece el cliente en función de la ronda
            cluster_index = ((server_round -1) // self.round_cluster) - 1
            # Seleccionar el grupo de clientes en función del índice del cluster
            clients = list(grupos.values())[cluster_index]

        # Return client/config pairs
        return [(client, fit_ins) for client, fit_ins in zip(clients, lista_fit_ins)]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        lista_eval_ins = []
        for conf_ in self.clusters:
            config_ = {"cluster": conf_}
            eval_ins = EvaluateIns(parameters, config_)
            lista_eval_ins.append(eval_ins)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if server_round > self.round_cluster:  # cambiar esto por cluster
            # Crear clusters
            grupos = {}
            for cluster, cliente in zip(self.clusters, clients):
                # Si el cluster aún no está en el diccionario, creamos una nueva lista para él
                if cluster not in grupos:
                    grupos[cluster] = []
                # Añadimos el cliente a la lista correspondiente al valor en el diccionario
                grupos[cluster].append(cliente)
            # if (server_round > self.round_cluster) and (server_round <= self.round_cluster * 2):
            #     clients = list(grupos.values())[0]  # selección del primer grupo
            # else:
            #     clients = list(grupos.values())[1]
            cluster_index = ((server_round-1) // self.round_cluster) - 1
            clients = list(grupos.values())[cluster_index] # selección del primer grupo

        # Return client/config pairs
        return [(client, evaluate_ins) for client, evaluate_ins in zip(clients, lista_eval_ins)]

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]

            aggregated_ndarrays, clusters = aggregate_fedmax(weights_results, self.clusters, server_round,
                                                             self.round_cluster)
        self.clusters = clusters
        # Save model
        if server_round <= self.round_cluster:
            print(f"Saving round {server_round} aggregated_parameters...")
            folder_path = 'saved_models'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_name = f'model_round_{server_round}.pth'
            file_path = os.path.join(folder_path, file_name)
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays) # Convert `List[np.ndarray]` to PyTorch`state_dict`
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)
            torch.save(net.state_dict(), file_path)
        # save model for printing results
        if results is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            folder_path = '.cache'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_name = f'model_round_fedmax{server_round}.pth'
            file_path = os.path.join(folder_path, file_name)

            # Save the model
            torch.save(net.state_dict(), file_path)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, EvaluateRes]],
            failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated