from flwr.server.strategy import Strategy
from flwr.common import FitRes, EvaluateRes, Parameters, Scalar, NDArrays
from typing import List, Optional, Tuple, Dict, Callable
import numpy as np
import flwr

class ClusterAvg(Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        initial_parameters: Optional[Parameters] = None,
    ):
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.evaluate_fn = evaluate_fn
        self.parameters = initial_parameters

    def initialize_parameters(self, client_manager):
        return self.parameters

    def configure_fit(self, server_round, parameters, client_manager):
        """Select clients and prepare fit instructions."""

        # Determine how many clients to select
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        fit_ins = flwr.common.FitIns(parameters, {})  # can add custom config as dict
        # Return list of tuples (client, fit_ins)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate clients parameters (weighted average)."""

        if not results:
            return self.parameters, {}

        weights_results = []
        num_examples_total = 0
        for client, fit_res in results:
            num_examples = fit_res.num_examples
            weights = flwr.common.parameters_to_ndarrays(fit_res.parameters)
            weights_results.append((weights, num_examples))
            num_examples_total += num_examples

        # Weighted average
        new_weights = [
            np.sum([w * n for w, n in zip(weights, [num for _, num in weights_results])], axis=0) / num_examples_total
            for weights, _ in weights_results[0:1]
        ]

        for i in range(1, len(weights_results)):
            for j, (w, n) in enumerate(zip(weights_results[i][0], [weights_results[i][1]])):
                new_weights[j] += w * n

        new_weights = [w / num_examples_total for w in new_weights]
        new_parameters = flwr.common.ndarrays_to_parameters(new_weights)
        self.parameters = new_parameters
        return new_parameters, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Select clients and prepare evaluation instructions."""
        sample_size, min_num_clients = self.num_evaluate_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        evaluate_ins = flwr.common.EvaluateIns(parameters, {})
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results (weighted average)."""
        if not results:
            return None, {}

        num_examples_total = sum([evaluate_res.num_examples for _, evaluate_res in results])
        weighted_loss = sum([evaluate_res.num_examples * evaluate_res.loss for _, evaluate_res in results])
        avg_loss = weighted_loss / num_examples_total
        return avg_loss, {}

    def evaluate(self, server_round, parameters):
        # Optional server evaluation
        if self.evaluate_fn is not None:
            return self.evaluate_fn(server_round, parameters, {})

        return None