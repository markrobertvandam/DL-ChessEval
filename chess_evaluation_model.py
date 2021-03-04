from tensorflow.keras import layers, models, Input
from tensorflow import keras
import numpy as np


class ChessEvaluationModel:
    def __init__(self, bitmap_shape: tuple, additional_features_shape: tuple) -> None:
        # Init model
        self.model = ChessEvaluationModel.__create_model(bitmap_shape, additional_features_shape)

    @staticmethod
    def __create_model(bitmap_shape, additional_features_shape) -> models.Model:
        # define the inputs
        input_cnn = Input(shape=bitmap_shape)
        input_numerical = Input(shape=additional_features_shape)
        # CNN
        conv_1 = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation="relu")(input_cnn)
        # Do we want max pooling?
        max_1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv_1)
        conv_2 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu")(max_1)
        # Do we want max pooling?
        max_2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_2)
        # Flatten data to allow concatenation with numerical feature vector
        flatten = layers.Flatten()(max_2)
        # Reduce the dimensionality before concatenating
        dense_1 = layers.Dense(100, activation="relu")(flatten)
        merged_layer = keras.layers.concatenate([dense_1, input_numerical])
        # Output evaluation of position
        output = layers.Dense(2, activation="linear")(merged_layer)

        return models.Model(inputs=[input_cnn, input_numerical], outputs=output)

    def get_summary(self) -> str:
        return self.model.summary()

    def compile(self, optimizer: str, loss: str) -> None:
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(
        self,
        train_data: list,  # list with 2 elements: [cnn_features, numerical_features]
        train_target: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> dict:
        return self.model.fit(
            train_data,
            train_target,
            epochs=epochs,
            validation_split=0.1,
            batch_size=batch_size
        )

    def train_validate(
        self,
        train_data: list,  # list with 2 elements: [cnn_features, numerical_features]
        train_target: np.ndarray,
        val_data: list,  # list with 2 elements: [cnn_features, numerical_features]
        val_target: np.ndarray,
        epochs: int,
        batch_size: int,
    ) -> dict:
        return self.model.fit(
            train_data,
            train_target,
            epochs=epochs,
            validation_data=(val_data, val_target),
            batch_size=batch_size
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
