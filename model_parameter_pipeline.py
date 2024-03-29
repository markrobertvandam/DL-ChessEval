from pathlib import Path
from typing import Dict

import numpy as np
from data_processing import DataProcessing
import tensorflow as tf
from chess_evaluation_model import ChessEvaluationModel


class ModelParameterPipeline:
    def __init__(
        self,
        bitmaps: np.ndarray,
        attributes: np.ndarray,
        labels: np.ndarray,
        plots_dir: Path,
        models_dir: Path,
        hyper_params: Dict,
    ):
        self.bitmaps = bitmaps
        self.attributes = attributes
        self.labels = labels
        self.train_bitmaps = None
        self.train_attributes = None
        self.train_labels = None
        self.test_bitmaps = None
        self.test_attributes = None
        self.test_labels = None
        # only path without name, name is going to be generated
        self.plots_dir = plots_dir
        # folder to save models in, allows multiple models to all save
        self.models_dir = models_dir
        self.hyper_params = hyper_params
        # example element: "activation_function":['sigmoid', 'relu', 'elu']

    def prepare_data(self):
        # split on train, val, test sets
        data_processing_obj = DataProcessing()
        (
            self.train_bitmaps,
            self.train_attributes,
            self.train_labels,
            self.test_bitmaps,
            self.test_attributes,
            self.test_labels,
        ) = data_processing_obj.train_test_split(
            self.bitmaps, self.attributes, self.labels
        )

    def run(
        self,
        learning_rate,
        activation_function,
        optimizer,
        dropout_rate,
        batch_size,
        epoch_number,
    ):
        if learning_rate is None:
            learning_rate = 0.01
        if activation_function is None:
            activation_function = "relu"
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()
        optimizer.learning_rate = learning_rate
        if dropout_rate is None:
            dropout_rate = 0.3
        if batch_size is None:
            batch_size = 512
        if epoch_number is None:
            epoch_number = 25

        # Initialize model
        chess_eval = ChessEvaluationModel()
        chess_eval.initialize(
            (8, 8, 12),
            (15,),
            optimizer,
            activation_function,
            dropout_rate,
            path_to_scalers=None,
        )

        history = chess_eval.train_validate(
            [self.train_bitmaps, self.train_attributes],
            [
                self.train_labels["eval"],
                self.train_labels["mate_turns"],
                self.train_labels["is_mate"],
            ],
            [self.test_bitmaps, self.test_attributes],
            [
                self.test_labels["eval"],
                self.test_labels["mate_turns"],
                self.test_labels["is_mate"],
            ],
            epoch_number,
            batch_size,
        )

        plot_name_eval = "af_{}_op_{}_dr_{}_bs_{}_ep_{}_lr_{}_eval.pdf".format(
            activation_function,
            optimizer._name,
            dropout_rate,
            batch_size,
            epoch_number,
            learning_rate,
        )

        plot_name_mate = "af_{}_op_{}_dr_{}_bs_{}_ep_{}_lr_{}_mate.pdf".format(
            activation_function,
            optimizer._name,
            dropout_rate,
            batch_size,
            epoch_number,
            learning_rate,
        )

        model_name = "af_{}_op_{}_dr_{}_bs_{}_ep_{}_lr_{}".format(
            activation_function,
            optimizer._name,
            dropout_rate,
            batch_size,
            epoch_number,
            learning_rate,
        )

        chess_eval.plot_history(
            history, self.plots_dir / plot_name_eval, type_loss="eval"
        )
        chess_eval.plot_history(
            history, self.plots_dir / plot_name_mate, type_loss="mate"
        )
        chess_eval.save_model(self.models_dir / model_name)

    def run_pipeline(self):

        self.prepare_data()

        for activation_function in self.hyper_params.get("activation_function", [None]):
            for optimizer in self.hyper_params.get("optimizer", [None]):
                for dropout_rate in self.hyper_params.get("dropout_rate", [None]):
                    for batch_size in self.hyper_params.get("batch_size", [None]):
                        for epoch_number in self.hyper_params.get(
                            "epoch_number", [None]
                        ):
                            for learning_rate in self.hyper_params.get(
                                "learning_rate", [None]
                            ):
                                self.run(
                                    learning_rate,
                                    activation_function,
                                    optimizer,
                                    dropout_rate,
                                    batch_size,
                                    epoch_number,
                                )
