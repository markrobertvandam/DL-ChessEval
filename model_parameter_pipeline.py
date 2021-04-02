import numpy as np
from data_processing import DataProcessing
import tensorflow as tf
from chess_evaluation_model import ChessEvaluationModel
import os


class ModelParameterPipeline:
    def __init__(
        self, bitmaps, attributes, labels, plot_path, save_path, dict_of_parameters
    ):
        self.bitmaps = bitmaps
        self.attributes = attributes
        self.labels = labels
        self.plot_path = (
            plot_path  # only path without name, name is going to be generated
        )
        self.save_path = (
            save_path  # folder to save models in, allows multiple models to all save
        )
        self.dict_of_parameters = dict_of_parameters
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

    def run_pipeline(self):

        self.prepare_data()

        for activation_function in self.dict_of_parameters.get(
            "activation_function", [None]
        ):
            for optimizer in self.dict_of_parameters.get("optimizer", [None]):
                for dropout_rate in self.dict_of_parameters.get("dropout_rate", [None]):
                    for batch_size in self.dict_of_parameters.get("batch_size", [None]):
                        for epoch_number in self.dict_of_parameters.get(
                            "epoch_number", [None]
                        ):
                            for learning_rate in self.dict_of_parameters.get(
                                "learning_rate", [None]
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
                                    path_to_scalers=None
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
                                        self.test_labels["is_mate"]
                                    ],
                                    epoch_number,
                                    batch_size,
                                )

                                plot_name = (
                                    "af_{}_op_{}_dr_{}_bs_{}_ep_{}_lr_{}.pdf".format(
                                        activation_function,
                                        optimizer._name,
                                        dropout_rate,
                                        batch_size,
                                        epoch_number,
                                        learning_rate,
                                    )
                                )

                                model_name = (
                                    "af_{}_op_{}_dr_{}_bs_{}_ep_{}_lr_{}".format(
                                        activation_function,
                                        optimizer._name,
                                        dropout_rate,
                                        batch_size,
                                        epoch_number,
                                        learning_rate,
                                    )
                                )

                                chess_eval.plot_history(
                                    history, os.path.join(self.plot_path, plot_name)
                                )

                                chess_eval.save_model(
                                    os.path.join(self.save_path, model_name)
                                )
