from tensorflow.keras import layers, models, Input
from tensorflow import keras
import numpy as np


class ChessEvaluationModel:
    def __init__(self, bitmap_shape: tuple, additional_features_shape: tuple) -> None:
        # Init model
        self.model = ChessEvaluationModel.__create_model(
            bitmap_shape, additional_features_shape
        )

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
        dropout_1 = layers.Dropout(0.3)(batch_norm_1)

        # Do we want max pooling? - FOR NOW, NO as we want to preserve the whole information
        # max_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_1)

        conv_2 = layers.Conv2D(
            50, kernel_size=(3, 3), strides=(1, 1), activation="elu"
        )(dropout_1)
        batch_norm_2 = layers.BatchNormalization()(conv_2)
        dropout_2 = layers.Dropout(0.3)(batch_norm_2)

        # Do we want max pooling? - FOR NOW, NO as we want to preserve the whole information
        # max_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

        # Flatten data to allow concatenation with numerical feature vector
        flatten = layers.Flatten()(dropout_2)
        # Reduce the dimensionality before concatenating
        dense_1 = layers.Dense(500, activation="elu")(flatten)
        merged_layer = keras.layers.concatenate([dense_1, input_numerical])
        # batch_norm_3 = layers.BatchNormalization()(merged_layer)
        # dropout_3 = layers.Dropout(0.3)(batch_norm_3)

        # Output evaluation of position
        output_eval = layers.Dense(1, activation="linear")(merged_layer)
        # Output number of turns to forced mate
        output_mate = layers.Dense(1, activation="linear")(merged_layer)
        # Output binary representing eval (0) or mate (1)
        output_binary = layers.Dense(1, activation="sigmoid")(merged_layer)

        return models.Model(
            inputs=[input_cnn, input_numerical],
            outputs=[output_eval, output_mate, output_binary],
        )

    def get_summary(self) -> str:
        return self.model.summary()

    def compile(
        self,
        optimizer: str = "SGD",
        loss: str = "root_mean_squared_error",
        metrics: list = [
            "root_mean_squared_error"
        ],  # list of metrics to evaluate model
    ) -> None:
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(
        self,
        train_data: list,  # list with 2 elements: [cnn_features, additional_features]
        train_target: list,  # list with 3 elements: [position_eval, num_turns_to_mate, binary for eval (0) or mate (1)]
        epochs: int = 100,
        batch_size: int = 128,
    ) -> dict:
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
        epochs: int,
        batch_size: int,
    ) -> dict:
        return self.model.fit(
            train_data,
            train_target,
            epochs=epochs,
            validation_data=(val_data, val_target),
            batch_size=batch_size,
        )


# Testing with CIFAR-10 dataset and random numerical features
"""
numerical_features_train = np.random.random((50000,5))
numerical_features_val = np.random.random((10000,5))
(train_images, train_labels), (val_images, val_labels) = datasets.cifar10.load_data()

model = Model((32,32,3), (5,))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
train_history = model.train(train_data=[train_images, numerical_features_train], train_target=train_labels,
            val_data=[val_images, numerical_features_val], val_target=val_labels, epochs=10)
"""
