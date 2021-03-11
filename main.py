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
    parser.add_argument("--scalers", help="Path to scalers")
    args = parser.parse_args()

    bitmaps = np.load(args.bitmaps)
    n_samples = round(len(bitmaps) * 0.01)
    bitmaps = bitmaps[:n_samples]
    attributes = np.load(args.attributes)[:n_samples]
    labels = np.load(args.labels)[:n_samples]
    path_to_scalers = args.scalers

    target_eval = labels["eval"][:, 0]
    target_mate = labels["eval"][:, 1]
    target_is_mate = labels["eval"][:, 2]

    chess_eval = ChessEvaluationModel()
    chess_eval.initialize(
        (8, 8, 12),
        (15,),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["mean_squared_error", "binary_accuracy"],
        path_to_scalers=path_to_scalers,
    )

    train_bitmaps = bitmaps[: int(len(bitmaps) * 0.9)]
    val_bitmaps = bitmaps[int(len(bitmaps) * 0.9) :]
    train_attributes = attributes[: int(len(attributes) * 0.9)]
    val_attributes = attributes[int(len(attributes) * 0.9) :]

    train_target_eval = target_eval[: int(len(target_eval) * 0.9)]
    val_target_eval = target_eval[int(len(target_eval) * 0.9) :]
    train_target_mate = target_mate[: int(len(target_mate) * 0.9)]
    val_target_mate = target_mate[int(len(target_mate) * 0.9) :]
    train_target_is_mate = target_is_mate[: int(len(target_is_mate) * 0.9)]
    val_target_is_mate = target_is_mate[int(len(target_is_mate) * 0.9) :]

    history = chess_eval.train_validate(
        [train_bitmaps, train_attributes],
        [train_target_eval, train_target_mate, train_target_is_mate],
        [val_bitmaps, val_attributes],
        [val_target_eval, val_target_mate, val_target_is_mate],
        10,
        256,
    )
    chess_eval.plot_history(
        history,
        args.plot.format(
            "eval_loss_0.1_bn_dropout_0.5_no_last_layer_normalized_target_test_scalers_class"
        ),
    )


if __name__ == "__main__":
    main()
