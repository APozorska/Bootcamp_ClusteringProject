
import logging
import json

from pathlib import Path

from clustering.data.processors import MSNBCDataProcessor
from clustering.models.kmeans import WebBehaviorClustering

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)


def main():
    # Initialize components
    processor = MSNBCDataProcessor()
    clusterer = WebBehaviorClustering()
    # visualizer = ClusterVisualizer()

    logger.info("Loading and preprocessing data...")
    # TODO: Load paths and parameters from some yaml config
    sequences = processor.load_data()
    logger.info(f"There are {len(sequences)} number of obs")
    X = processor.preprocess_sequences(sequences)[:10000, :]
    logger.info(f"There are {X.shape[0]} obs after filtering")

    # Find optimal number of clusters
    logger.info("Finding optimial number of clusters...")
    cluster_scores = clusterer.find_optimal_n_clusters(X)

    # Fit model
    logger.info("Fitting clustering model after optimization...")
    clusterer.fit(X)

    # Evaluate the results
    logger.info("Evaluating the model...")
    metrics = clusterer.evaluate_clustering(X)
    logger.info(f"Clustering metrics: {metrics}")

    # Save results
    results = {
        "metrics": metrics,
        "cluster_scores": [(int(k), float(v)) for k, v in cluster_scores],
        "best_params": clusterer.best_params_,
    }
    results_path = Path("results")
    results_path.mkdir(exist_ok=True, parents=True)
    with (results_path / "msnbc_clustering_results.json").open(mode="w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

