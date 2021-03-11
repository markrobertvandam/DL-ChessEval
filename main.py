import argparse

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf

from chess_evaluation_model import ChessEvaluationModel
from data_processing import DataProcessing


def main():
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    parser.add_argument("--bitmaps", help="Path to the input bitmaps")
    parser.add_argument("--attributes", help="Path to the additional input attributes")
    parser.add_argument("--labels", help="Path to the input labels")
    parser.add_argument("--plot", help="Path to the generated history plot image")
    parser.add_argument("--scalers", help="Path to scalers")
    args = parser.parse_args()

    # load all data and path to scalers if given
    bitmaps = np.load(args.bitmaps)
    n_samples = round(len(bitmaps) * 0.01)
    bitmaps = bitmaps[:n_samples]
    attributes = np.load(args.attributes)[:n_samples]
    labels = np.load(args.labels)[:n_samples]
    path_to_scalers = args.scalers

    # split on train, val, test sets
    data_processing_obj = DataProcessing()
    (
        train_bitmaps,
        train_attributes,
        train_labels,
        val_bitmaps,
        val_attributes,
        val_labels,
        test_bitmaps,
        test_attributes,
        test_labels,
    ) = data_processing_obj.train_val_test_split(bitmaps, attributes, labels)

    train_target_eval = train_labels["eval"][:, 0]
    train_target_mate = train_labels["eval"][:, 1]
    train_target_is_mate = train_labels["eval"][:, 2]

    val_target_eval = val_labels["eval"][:, 0]
    val_target_mate = val_labels["eval"][:, 1]
    val_target_is_mate = val_labels["eval"][:, 2]

    test_target_eval = test_labels["eval"][:, 0]
    test_target_mate = test_labels["eval"][:, 1]
    test_target_is_mate = test_labels["eval"][:, 2]

    # init model and train
    chess_eval = ChessEvaluationModel()
    chess_eval.initialize(
        (8, 8, 12),
        (15,),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=["mean_squared_error", "binary_accuracy"],
        path_to_scalers=path_to_scalers,
    )

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

    mse_eval, mse_mate, accuracy_is_mate = chess_eval.get_mse_inverse_transform(
        [test_bitmaps, test_attributes],
        [test_target_eval, test_target_mate, test_target_is_mate],
    )

    print("MSE on inverse test eval: {}".format(mse_eval))
    print("MSE on inverse test mate: {}".format(mse_mate))
    print("Accuracy on test is_mate: {}".format(accuracy_is_mate))


if __name__ == "__main__":
    main()
