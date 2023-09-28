from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


#strategy = AggregateCustomMetricStrategy(fraction_fit=0.5, fraction_eval=0.5,min_fit_clients=2,
#        min_eval_clients=2,
#        min_available_clients=2,)




# Define strategy
# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,min_available_clients=10)
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average,min_available_clients=2)


# Start Flower server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
