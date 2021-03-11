import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models, Input
from tensorflow.python.keras.callbacks import History
from data_processing import DataProcessing
from sklearn.metrics import mean_squared_error, accuracy_score


class ChessEvaluationModel:
    def __init__(self):
        self.model = None
        self.data_processing_obj = DataProcessing()

    @staticmethod
    def __create_model(bitmap_shape, additional_features_shape) -> models.Model:
        # define the inputs
        input_cnn = Input(shape=bitmap_shape)
        input_numerical = Input(shape=additional_features_shape)

        # Marco Wiering paper architecture
        conv_1 = layers.Conv2D(
            20, kernel_size=(5, 5), strides=(1, 1), activation="elu"
        )(input_cnn)
        batch_norm_1 = layers.BatchNormalization()(conv_1)
        dropout_1 = layers.Dropout(0.5)(batch_norm_1)

        # Do we want max pooling? - FOR NOW, NO as we want to preserve the whole information
        # max_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_1)

        conv_2 = layers.Conv2D(
            50, kernel_size=(3, 3), strides=(1, 1), activation="elu"
        )(dropout_1)
        batch_norm_2 = layers.BatchNormalization()(conv_2)
        dropout_2 = layers.Dropout(0.5)(batch_norm_2)

        # Do we want max pooling? - FOR NOW, NO as we want to preserve the whole information
        # max_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

        # Flatten data to allow concatenation with numerical feature vector
        flatten = layers.Flatten()(dropout_2)
        # Reduce the dimensionality before concatenating
        dense_1 = layers.Dense(1000, activation="elu")(flatten)
        merged_layer = keras.layers.concatenate([dense_1, input_numerical])
        batch_norm_3 = layers.BatchNormalization()(merged_layer)
        # dropout_3 = layers.Dropout(0.5)(batch_norm_3)

        # Output evaluation of position
        output_eval = layers.Dense(1, activation="linear", name="output_eval")(
            batch_norm_3
        )
        # Output number of turns to forced mate
        output_mate = layers.Dense(1, activation="linear", name="output_mate")(
            batch_norm_3
        )
        # Output binary representing eval (0) or mate (1)
        output_binary = layers.Dense(1, activation="sigmoid", name="output_binary")(
            batch_norm_3
        )

        return models.Model(
            inputs=[input_cnn, input_numerical],
            outputs=[output_eval, output_mate, output_binary],
        )

    @staticmethod
    def plot_history(history, plot_path: str):
        history = history.history
        # plot the training loss and accuracy
        n = np.arange(0, len(history["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(n, history["loss"], label="overall_train_loss")
        plt.plot(n, history["output_eval_loss"], label="train_eval_loss")
        plt.plot(n, history["output_mate_loss"], label="train_mate_loss")
        plt.plot(
            n, history["output_binary_binary_accuracy"], label="train_is_mate_accuracy"
        )
        if "val_loss" in history:
            plt.plot(n, history["val_loss"], label="overall_val_loss")
            plt.plot(n, history["val_output_eval_loss"], label="val_eval_loss")
            plt.plot(n, history["val_output_mate_loss"], label="val_mate_loss")
            plt.plot(
                n,
                history["val_output_binary_binary_accuracy"],
                label="val_is_mate_accuracy",
            )
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(plot_path)

    def get_summary(self) -> str:
        return self.model.summary()

    def initialize(
        self,
        bitmap_shape: tuple,
        additional_features_shape: tuple,
        optimizer: str = "SGD",
        loss: dict = {
            "output_eval": "mean_squared_error",
            "output_mate": "mean_squared_error",
            "output_binary": "binary_crossentropy",
        },  # Dict with key the name of the output layer and value the loss function
        loss_weights: list = None,  # We can specify different weight for each loss
        metrics: list = None,  # list of metrics to evaluate model
        path_to_scalers: str = None,  # path to scalers
    ) -> None:

        self.model = self.__create_model(bitmap_shape, additional_features_shape)
        self.model.compile(
            optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights
        )

        # Initialize or load scalers
        if path_to_scalers is None:
            self.data_processing_obj.init_scalers()
        else:
            self.data_processing_obj.load_scalers(path=path_to_scalers)

    def train_redundant(
        self,
        train_data: list,  # list with 2 elements: [cnn_features, additional_features]
        train_target: list,  # list with 3 elements: [position_eval, num_turns_to_mate, binary for eval (0) or mate (1)]
        epochs: int = 100,
        batch_size: int = 128,
    ) -> History:
        return self.model.fit(
            train_data,
            train_target,
            epochs=epochs,
            validation_split=0.1,
            batch_size=batch_size,
        )

    def train_validate(
        self,
        train_data: list,  # list with 2 elements: [cnn_features, additional_features]
        train_target: list,  # list with 3 elements: [position_eval, num_turns_to_mate, binary for eval (0) or mate (1)]
        val_data: list,  # list with 2 elements: [cnn_features, additional_features]
        val_target: list,  # list with 3 elements: [position_eval, num_turns_to_mate, binary for eval (0) or mate (1)]
        epochs: int = 100,
        batch_size: int = 128,
    ) -> dict:

        train_eval_reshaped = train_target[0].reshape(-1, 1)
        train_mate_reshaped = train_target[1].reshape(-1, 1)

        val_eval_reshaped = val_target[0].reshape(-1, 1)
        val_mate_reshaped = val_target[1].reshape(-1, 1)

        # fit the training targets for eval and mate
        self.data_processing_obj.fit_scalers(train_eval_reshaped, train_mate_reshaped)

        # transform train targets
        train_eval_normalized = self.data_processing_obj.transform(
            train_eval_reshaped, data_type="eval"
        )
        train_mate_normalized = self.data_processing_obj.transform(
            train_mate_reshaped, data_type="mate"
        )

        # transform val targets
        val_eval_normalized = self.data_processing_obj.transform(
            val_eval_reshaped, data_type="eval"
        )
        val_mate_normalized = self.data_processing_obj.transform(
            val_mate_reshaped, data_type="mate"
        )

        train_target = [train_eval_normalized, train_mate_normalized, train_target[2]]
        val_target = [val_eval_normalized, val_mate_normalized, val_target[2]]

        return self.model.fit(
            train_data,
            train_target,
            epochs=epochs,
            validation_data=(val_data, val_target),
            batch_size=batch_size,
        )

    def test(self, test_data, test_target, batch_size: int = 128):
        # reshape data so we can transform
        test_eval_reshaped = test_target[0].reshape(-1, 1)
        test_mate_reshaped = test_target[1].reshape(-1, 1)

        # transform test targets
        test_eval_normalized = self.data_processing_obj.transform(
            test_eval_reshaped, data_type="eval"
        )
        test_mate_normalized = self.data_processing_obj.transform(
            test_mate_reshaped, data_type="mate"
        )

        test_target = [test_eval_normalized, test_mate_normalized, test_target[2]]

        return self.model.evaluate(test_data, test_target, batch_size=batch_size)

    def get_mse_inverse_transform(self, test_data, test_target):
        predictions = self.predict(test_data)

        eval_predictions_inversed = self.data_processing_obj.inverse_transform(
            predictions[0], data_type="eval"
        )
        mate_predictions_inversed = self.data_processing_obj.inverse_transform(
            predictions[1], data_type="mate"
        )
        is_mate_predictions = (predictions[2] > 0.5).astype(int)

        mse_eval = mean_squared_error(eval_predictions_inversed, test_target[0])
        mse_mate = mean_squared_error(mate_predictions_inversed, test_target[1])
        accuracy_is_mate = accuracy_score(is_mate_predictions, test_target[2])

        return mse_eval, mse_mate, accuracy_is_mate

    def predict(self, data):
        return self.model.predict(data)