from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


# Define metric aggregation function
def weighted_accuracy_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_f1_score(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    f1_scores = [num_examples * m["f1 score"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"f1 score": sum(f1_scores) / sum(examples)}


# Define strategy 
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,min_available_clients=10)
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_f1_score,min_available_clients=2)
# fl.server.strategy.FedAdagrad()
# Start Flower server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
