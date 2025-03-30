from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline


class MSNBCDataProcessor:
    def __init__(self):
        self.category_mapping = {
            1: 'frontpage', 2: 'news', 3: 'tech', 4: 'local',
            5: 'opinion', 6: 'on-air', 7: 'misc', 8: 'weather',
            9: 'health', 10: 'living', 11: 'business', 12: 'sports',
            13: 'summary', 14: 'bbs', 15: 'travel', 16: 'msn-news',
            17: 'msn-sports'
        }
        self.data_path: Path | None = None
        self.preprocessor = MSNBCDataProcessor.create_preprocessor()

    def load_data(self, file_path: str | Path = "data/msnbc.seq") -> list[list[int]]:
        sequences: list[list[int]] = []
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"The given path {file_path} to data does not exist!")

        self.data_path = file_path
        with file_path.open() as f:
            for line in f:
                if not line.startswith("%"):
                    # Convert all lines to seqences
                    sequence = [int(x) for x in line.strip().split()]
                    if line.strip():
                        sequences.append(sequence)
        return sequences

    # zliczanie liczby wystapien w danej sekwencji
    @staticmethod
    def create_feature_vector(sequence: list[int]):
        counts = Counter(sequence)
        return [counts.get(i, 0) for i in range(1, 18)]

    @staticmethod
    def create_preprocessor():
        frequency_transformer = Pipeline([
            ("scaler", StandardScaler()),
            ("normalize", PowerTransformer(method="yeo-johnson"))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("frequencies", frequency_transformer, list(range(17)))
            ],
            remainder="passthrough"
        )
        return preprocessor

    def preprocess_sequences(self, sequences: list[list[int]]):
        feature_vectors = [
            self.create_feature_vector(seq) for seq in sequences
        ]
        X = np.array(feature_vectors)
        X_transformed = self.preprocessor.fit_transform(X)
        return X_transformed

    def get_feature_names(self):
        return [self.category_mapping[i] for i in range(1, 18)]


