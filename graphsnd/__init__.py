"""Graph-SND: scalable System Neural Diversity via graph-based local
aggregation and sampling estimators."""

from graphsnd.graphs import (
    bernoulli_edges,
    complete_edges,
    knn_edges,
    uniform_size_edges,
)
from graphsnd.metrics import (
    graph_snd,
    graph_snd_from_rollouts,
    hoeffding_bound,
    ht_estimator,
    pairwise_behavioral_distance,
    pairwise_distances_on_edges,
    serfling_bound,
    snd,
    uniform_sample_estimator,
)
from graphsnd.wasserstein import (
    wasserstein_gaussian,
    wasserstein_gaussian_diag,
)

__version__ = "0.1.0"

__all__ = [
    "bernoulli_edges",
    "complete_edges",
    "graph_snd",
    "graph_snd_from_rollouts",
    "hoeffding_bound",
    "ht_estimator",
    "knn_edges",
    "pairwise_behavioral_distance",
    "pairwise_distances_on_edges",
    "serfling_bound",
    "snd",
    "uniform_sample_estimator",
    "uniform_size_edges",
    "wasserstein_gaussian",
    "wasserstein_gaussian_diag",
]
