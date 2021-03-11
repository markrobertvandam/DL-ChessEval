from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pickle
import os


class DataProcessing:
    def __init__(
        self, scaler_type_eval: str = "min_max", scaler_type_mate="min_max"
    ) -> None:
        self.scaler_type_eval = scaler_type_eval
        self.scaler_type_mate = scaler_type_mate

        self.scaler_eval = None
        self.scaler_mate = None

    def init_scalers(self) -> None:
        if self.scaler_type_eval == "min_max":
            self.scaler_eval = MinMaxScaler()
        elif self.scaler_type_eval == "standard":
            self.scaler_eval = StandardScaler()
        if self.scaler_type_mate == "min_max":
            self.scaler_mate = MinMaxScaler()
        elif self.scaler_type_mate == "standard":
            self.scaler_mate = StandardScaler()

    def fit_scalers(
        self, train_eval_target: np.ndarray, train_mate_target: np.ndarray
    ) -> None:
        self.scaler_eval.fit(train_eval_target)
        self.scaler_mate.fit(train_mate_target)

    def transform(self, target_data, data_type: str = "eval") -> np.ndarray:
        if data_type == "eval":
            return self.scaler_eval.transform(target_data)
        elif data_type == "mate":
            return self.scaler_mate.transform(target_data)

    def inverse_transform(self, target_data, data_type: str = "eval") -> np.ndarray:
        if data_type == "eval":
            return self.scaler_eval.inverse_transform(target_data)
        elif data_type == "mate":
            return self.scaler_mate.inverse_transform(target_data)

    def get_scalers(self):
        return self.scaler_eval, self.scaler_mate

    def save_scalers(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)

        # save scalers
        pickle.dump(self.scaler_eval, open(os.path.join(path, "scaler_eval.pkl"), "wb"))
        pickle.dump(self.scaler_mate, open(os.path.join(path, "scaler_mate.pkl"), "wb"))

    def load_scalers(self, path: str) -> None:
        try:
            self.scaler_eval = pickle.load(
                self.scaler_eval,
                open(os.path.join(path, "scaler_eval.pkl"), "rb"),
            )
            self.scaler_mate = pickle.load(
                self.scaler_mate,
                open(os.path.join(path, "scaler_mate.pkl"), "rb"),
            )
        except Exception as e:
            print(e)
            print("No scalers found in the given path.")
