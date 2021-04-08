from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pickle


class DataProcessing:
    def __init__(self, scaler_type_eval: str = "min_max") -> None:
        self.scaler_type_eval = scaler_type_eval
        self.scaler_eval = None

    def init_scalers(self) -> None:
        if self.scaler_type_eval == "min_max":
            self.scaler_eval = MinMaxScaler()
        elif self.scaler_type_eval == "standard":
            self.scaler_eval = StandardScaler()

    def fit_scalers(self, train_eval_target: np.ndarray) -> None:
        self.scaler_eval.fit(train_eval_target)

    def transform(self, target_data: np.ndarray) -> np.ndarray:
        return self.scaler_eval.transform(target_data)

    def inverse_transform(self, target_data: np.ndarray) -> np.ndarray:
        return self.scaler_eval.inverse_transform(target_data)

    def get_scalers(self):
        return self.scaler_eval

    def save_scalers(self, path: Path) -> None:
        if not path.exists():
            Path.mkdir(path)

        # save scalers
        pickle.dump(self.scaler_eval, open(path / "scaler_eval.pkl", "wb"))

    def load_scalers(self, path: Path) -> None:
        try:
            self.scaler_eval = pickle.load(open(path / "scaler_eval.pkl", "rb"))
        except Exception as e:
            print(e)
            print("No scalers found in the given path.")

    @staticmethod
    def train_val_test_split(
        bitmaps: np.ndarray, attrs: np.ndarray, labels: np.ndarray,
        train_split: float = 0.8, val_split: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray,
               np.ndarray, np.ndarray, np.ndarray]:
        train_bitmaps, val_bitmaps, test_bitmaps = DataProcessing.split_logic(bitmaps, train_split, val_split)
        train_attrs, val_attrs, test_attrs = DataProcessing.split_logic(attrs, train_split, val_split)
        train_labels, val_labels, test_labels = DataProcessing.split_logic(labels, train_split, val_split)

        return (
            train_bitmaps,
            train_attrs,
            train_labels,
            val_bitmaps,
            val_attrs,
            val_labels,
            test_bitmaps,
            test_attrs,
            test_labels,
        )

    @staticmethod
    def train_test_split(
        bitmaps: np.ndarray, attrs: np.ndarray, labels: np.ndarray, train_split: float = 0.9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        output = DataProcessing.train_val_test_split(bitmaps, attrs, labels, train_split=train_split, val_split=0)
        return (
            output[0],
            output[1],
            output[2],
            output[6],
            output[7],
            output[8]
        )

    @staticmethod
    def split_logic(
        data: np.ndarray, train_split: float = 0.8, val_split: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_data = data[: int(len(data) * train_split)]
        val_data = data[int(len(data) * train_split): int(len(data) * (train_split + val_split))]
        test_data = data[int(len(data) * (train_split + val_split)):]

        return train_data, val_data, test_data
