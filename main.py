import argparse

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf

from chess_evaluation_model import ChessEvaluationModel
from model_parameter_pipeline import ModelParameterPipeline


def main():
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    parser.add_argument("--bitmaps", help="Path to the input bitmaps")
    parser.add_argument("--attributes", help="Path to the additional input attributes")
    parser.add_argument("--labels", help="Path to the input labels")
    parser.add_argument("--plot", help="Path to the generated history plot image")
    args = parser.parse_args()

    bitmaps = np.load(args.bitmaps)
    n_samples = round(len(bitmaps) * 0.01)
    bitmaps = bitmaps[:n_samples]
    attributes = np.load(args.attributes)[:n_samples]
    labels = np.load(args.labels)[:n_samples]

    target_eval = labels["eval"][:, 0]
    target_mate = labels["eval"][:, 1]
    target_is_mate = labels["eval"][:, 2]

    # Normalize labels
    scaler_eval = MinMaxScaler()
    scaler_eval.fit(target_eval.reshape(-1, 1))
    target_eval_normalized = scaler_eval.transform(target_eval.reshape(-1, 1))

    scaler_mate = MinMaxScaler()
    scaler_mate.fit(target_mate.reshape(-1, 1))
    target_mate_normalized = scaler_mate.transform(target_mate.reshape(-1, 1))

    chess_eval = ChessEvaluationModel()
    chess_eval.initialize(
        (8, 8, 12),
        (15,),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["mean_squared_error", "binary_accuracy"],
    )
    history = chess_eval.train(
        [bitmaps, attributes],
        [target_eval_normalized, target_mate_normalized, target_is_mate],
        10,
        256,
    )
    chess_eval.plot_history(
        history, args.plot.format("eval_loss_0.1_bn_dropout_0.5_no_last_layer_normalized_target")
    )


if __name__ == "__main__":
    main()
