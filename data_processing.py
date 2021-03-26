from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pickle
import os


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

    def transform(self, target_data) -> np.ndarray:
        return self.scaler_eval.transform(target_data)

    def inverse_transform(self, target_data) -> np.ndarray:
        return self.scaler_eval.inverse_transform(target_data)

    def get_scalers(self):
        return self.scaler_eval

    def save_scalers(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        # save scalers
        pickle.dump(self.scaler_eval, open(os.path.join(path, "scaler_eval.pkl"), "wb"))

    def load_scalers(self, path: str) -> None:
        try:
            self.scaler_eval = pickle.load(
                self.scaler_eval,
                open(os.path.join(path, "scaler_eval.pkl"), "rb"),
            )
        except Exception as e:
            print(e)
            print("No scalers found in the given path.")

    def train_val_test_split(
            self, bitmaps, attributes, labels, train_split=0.8, val_split=0.1
    ) -> tuple:
        train_bitmaps, val_bitmaps, test_bitmaps = self.split_logic(
            bitmaps, train_split, val_split
        )
        train_attributes, val_attributes, test_attributes = self.split_logic(
            attributes, train_split, val_split
        )
        train_labels, val_labels, test_labels = self.split_logic(
            labels, train_split, val_split
        )

        return (
            train_bitmaps,
            train_attributes,
            train_labels,
            val_bitmaps,
            val_attributes,
            val_labels,
            test_bitmaps,
            test_attributes,
            test_labels,
        )

    def split_logic(self, data, train_split=0.8, val_split=0.1) -> tuple:
        train_data = data[: int(len(data) * train_split)]
        val_data = data[int(len(data) * train_split): int(len(data) * (train_split + val_split))]
        test_data = data[int(len(data) * (train_split + val_split)):]

        return train_data, val_data, test_data
