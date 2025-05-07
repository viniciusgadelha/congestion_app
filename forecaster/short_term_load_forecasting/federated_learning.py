from short_term_load_forecasting import generate_client, weighted_average_fn

import flwr as fl

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.experimental import compose
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

def fed_learn(ids, params, plotting=False):
    # Define client
    client_fn = generate_client(ids, params)

    # Define strategy
    strategy = instantiate(params.strategy, evaluate_metrics_aggregation_fn=weighted_average_fn)


    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=params.num_clients,
        config=fl.server.ServerConfig(num_rounds=params.num_rounds),
        strategy=strategy,
        # client_manager=client_manager
        # client_resources=client_resources
    )
