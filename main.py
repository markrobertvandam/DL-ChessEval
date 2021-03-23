import argparse

import numpy as np
import tensorflow as tf

from chess_evaluation_model import ChessEvaluationModel
from data_processing import DataProcessing


def gpu_fix() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def main():
    gpu_fix()
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    parser.add_argument("-b", "--bitmaps", help="Path to the input bitmaps", required=True)
    parser.add_argument("-a", "--attributes", help="Path to the additional input attributes", required=True)
    parser.add_argument("-l", "--labels", help="Path to the input labels", required=True)
    parser.add_argument("-p", "--plot", help="Path to the generated history plot image")
    parser.add_argument("-s", "--scalers", help="Path to scalers")
    parser.add_argument("-m", "--model", help="Path to created model")
    args = parser.parse_args()

    # load all data and path to scalers if given
    bitmaps = np.load(args.bitmaps)
    n_samples = round(len(bitmaps) * 0.1)
    bitmaps = bitmaps[n_samples:2*n_samples]
    attributes = np.load(args.attributes)[n_samples:2*n_samples]
    labels = np.load(args.labels)[n_samples:2*n_samples]
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
        path_to_scalers=path_to_scalers,
    )

    history = chess_eval.train_validate(
        [train_bitmaps, train_attributes],
        [train_target_eval, train_target_mate, train_target_is_mate],
        [val_bitmaps, val_attributes],
        [val_target_eval, val_target_mate, val_target_is_mate],
        15,
        512,
    )

    if args.model is not None:
        chess_eval.save_model(args.model)

    chess_eval.plot_history(
        history,
        args.plot.format(
            "eval_loss_0.1_bn_dropout_0.3_normalized_target_test_scalers_class"
        ),
    )

    mse_eval, mse_mate, accuracy_is_mate = chess_eval.get_mse_inverse_transform(
        [test_bitmaps, test_attributes],
        [test_target_eval, test_target_mate, test_target_is_mate],
    )

    print("RMS on inverse test eval: {:7.2f}".format(np.sqrt(mse_eval)))
    print("RMS on inverse test mate: {:7.4f}".format(np.sqrt(mse_mate)))
    print("Accuracy on test is_mate: {:7.4f}".format(accuracy_is_mate))


if __name__ == "__main__":
    main()
