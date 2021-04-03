import argparse
from typing import Tuple

import numpy as np

from data_processing import DataProcessing


def gpu_fix() -> None:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    # parser.add_argument("-sc", "--scalers", help="Path to scalers")

    sp = parser.add_subparsers(title="Command", dest="command")
    shared = argparse.ArgumentParser(description="Shared parameters", add_help=False)
    shared.add_argument(help="Directory where input files are saved")
    shared.add_argument("-d", "--data", "bitmaps", help="Path to the input bitmaps")
    shared.add_argument("attributes", help="Path to the additional input attributes")
    shared.add_argument("labels", help="Path to the input labels")

    tune_run = sp.add_parser("pipeline", parents=[shared], help="Tuning run using the pipeline")
    tune_run.add_argument("models", help="Directory where created models are saved")
    tune_run.add_argument("plots", help="Directory where generated history plots are saved")
    tune_run.add_argument("percentage", type=int, help="Percentage of data to use")
    tune_run.add_argument("-o", "--offset", type=int, default=0,
                          help="Offset from start of data to take percentage from")

    test_run = sp.add_parser("test", parents=[shared], help="Test run with a previously saved model")
    test_run.add_argument("model", help="Path to previously saved model")
    test_run.add_argument("percentage", type=int, help="Percentage of data to use")
    test_run.add_argument("-o", "--offset", type=int, default=0,
                          help="Offset from start of data to take percentage from")

    full = sp.add_parser("full-run", parents=[shared], help="Do a run with all the data")
    full.add_argument("model", help="Directory where created model is saved")
    full.add_argument("plot", help="Path and filename where history plot is saved")

    return parser.parse_args()


def slice_data(
        e_bitmaps_path, e_attributes_path, e_labels_path, m_bitmaps_path,
        m_attributes_path, m_labels_path, percentage, offset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bitmap_evals = np.load(e_bitmaps_path)
    n_samples = round(len(bitmap_evals) * percentage / 100)
    offset = round(len(bitmap_evals) * offset / 100)

    bitmap_mates = np.load(m_bitmaps_path)
    n_mates = round(len(bitmap_mates) * percentage / 100)
    offset_mates = round(len(bitmap_mates) * offset / 100)
    
    bitmaps = np.concatenate(bitmap_evals[offset:offset + n_samples], bitmap_mates[offset_mates:offset_mates + n_mates])

    return (
        bitmaps,
        np.concatenate(
            np.load(e_attributes_path)[offset:offset + n_samples],
            np.load(m_attributes_path)[offset_mates:offset_mates + n_mates]
        ),
        np.concatenate(
            np.load(e_labels_path)[offset:offset + n_samples],
            np.load(m_labels_path)[offset_mates:offset_mates + n_mates]
        )
    )


def tune(args: argparse.Namespace) -> None:
    from model_parameter_pipeline import ModelParameterPipeline
    e_bitmaps_path = args.eb if args.eb else args.data + "eval_bitmaps.npy"
    e_attributes_path = args.ea if args.ea else args.data + "eval_attributes.npy"
    e_labels_path = args.el if args.el else args.data + "eval_labels.npy"
    m_bitmaps_path = args.mb if args.mb else args.data + "mate_bitmaps.npy"
    m_attributes_path = args.ma if args.ma else args.data + "mate_attributes.npy"
    m_labels_path = args.ml if args.ml else args.data + "mate_labels.npy"
    bitmaps, attributes, labels = slice_data(e_bitmaps_path, e_attributes_path, e_labels_path,
                                             m_bitmaps_path, m_attributes_path, m_labels_path,
                                             args.percentage, args.offset)

    # Test parameter pipeline
    dict_of_params = {
        # "batch_size": [256, 512],
        # "activation_function": ["relu", "elu"],
        # "dropout_rate": [0.3, 0.5],
        # "epoch_number": [50]
    }

    model_param_pipeline = ModelParameterPipeline(
        bitmaps, attributes, labels, args.plots, args.models, dict_of_params
    )
    model_param_pipeline.run_pipeline()


def test(args: argparse.Namespace) -> None:
    from chess_evaluation_model import ChessEvaluationModel
    bitmaps, attributes, labels = slice_data(args.bitmaps, args.attributes, args.labels, args.percentage, args.offset)

    chess_eval = ChessEvaluationModel()
    chess_eval.load_model(args.model)
    chess_eval.test([bitmaps, attributes], labels)


def full_run(args: argparse.Namespace):
    from chess_evaluation_model import ChessEvaluationModel
    bitmaps = np.load(args.bitmaps)
    attributes = np.load(args.attributes)
    labels = np.load(args.labels)

    chess_eval = ChessEvaluationModel()
    chess_eval.initialize((8, 8, 12), (15,), "Adam", "relu", 0.3)

    data_processing_obj = DataProcessing()
    (
        train_bitmaps,
        train_attributes,
        train_labels,
        test_bitmaps,
        test_attributes,
        test_labels,
    ) = data_processing_obj.train_test_split(
        bitmaps, attributes, labels
    )

    history = chess_eval.train_validate(
        [train_bitmaps, train_attributes],
        [train_labels["eval"], train_labels["mate_turns"], train_labels["is_mate"]],
        [test_bitmaps, test_attributes],
        [test_labels["eval"], test_labels["mate_turns"], test_labels["is_mate"]],
        50,
        512,
    )

    chess_eval.save_model(args.model)
    chess_eval.plot_history(
        history,
        args.plot,
    )


def main():
    args = parse_args()
    gpu_fix()
    print(args)
    exit()

    if args.command == "pipeline":
        tune(args)
    elif args.command == "test":
        test(args)
    elif args.command == "full-run":
        full_run(args)


if __name__ == "__main__":
    main()
