import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score


class WebBehaviorClustering:
    def __init__(self, n_clusters: int = 8, batch_size: int = 128, random_state: int = 42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.best_params_ = None

    def _create_model(self, n_clusters: int):
        return MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=self.batch_size,
            random_state=self.random_state,
        )

    def find_optimal_n_clusters(self, X: np.array, cluster_range=range(2,18)) -> list[tuple[int, float]]:
        scores: list[tuple[int, float]] = []
        for n_clusters in tqdm(cluster_range):
            model = self._create_model(n_clusters)
            labels = model.fit_predict(X)
            score = silhouette_score(X, labels)
            scores.append((n_clusters, score))

        self.best_params_ = max(scores, key=lambda x: x[1])
        # porownujemy maksimum po scorach - drugi element z tupli
        return scores

    def fit(self, X: np.array):
        if self.best_params_ is None:
            self.model = self._create_model(self.n_clusters)
        else:
            self.model = self._create_model(self.best_params_[0])
        self.model.fit(X)
        return self

    def get_cluster_centerss(self):
        if self.model is not None:
            return self.model.cluster_centers_

    def get_labels(self, X):
        if self.model is not None:
            return self.model.predict(X)

    def evaluate_clustering(self, X: np.array):
        labels = self.get_labels(X)
        if labels is not None:
            metrics = {
                "silhouette_score": silhouette_score(X, labels),
                "sse": self.model.inertia_,
            }
        return metrics