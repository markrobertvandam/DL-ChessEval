#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


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
    parser = argparse.ArgumentParser(description="Run the Chess Evaluation Model")
    # parser.add_argument("-sc", "--scalers", help="Path to scalers")

    # Params that are shared between all commands
    shared = argparse.ArgumentParser(description="Shared parameters", add_help=False)
    shared.add_argument("data", type=Path, help="Input files directory")
    shared.add_argument(
        "-eb",
        "--eval-bitmaps",
        default="eval_bitmaps.npy",
        help="Override eval input bitmap filename",
    )
    shared.add_argument(
        "-ea",
        "--eval-attributes",
        default="eval_attributes.npy",
        help="Override additional eval input attributes filename",
    )
    shared.add_argument(
        "-el",
        "--eval-labels",
        default="eval_labels.npy",
        help="Override eval input labels filename",
    )
    shared.add_argument(
        "-mb",
        "--mate-bitmaps",
        default="mate_bitmaps.npy",
        help="Override mate input bitmap filename",
    )
    shared.add_argument(
        "-ma",
        "--mate-attributes",
        default="mate_attributes.npy",
        help="Override additional mate input attributes filename",
    )
    shared.add_argument(
        "-ml",
        "--mate-labels",
        default="mate_labels.npy",
        help="Override mate input labels filename",
    )

    tune_test_shared = argparse.ArgumentParser(
        description="Shared parameters of pipeline and test", add_help=False
    )
    tune_test_shared.add_argument(
        "percentage", type=int, help="Percentage of data to use"
    )
    tune_test_shared.add_argument(
        "-o",
        "--offset",
        type=int,
        default=0,
        help="Offset from start of data to take percentage from",
    )

    # Create a sub-parser group for each command
    sp = parser.add_subparsers(title="Command", dest="command")

    tune_run = sp.add_parser(
        "pipeline",
        parents=[shared, tune_test_shared],
        help="Tuning run using the pipeline",
    )
    tune_run.add_argument(
        "models", type=Path, help="Directory where created models will be saved"
    )
    tune_run.add_argument(
        "plots", type=Path, help="Directory where generated history plots will be saved"
    )

    test_run = sp.add_parser(
        "test",
        parents=[shared, tune_test_shared],
        help="Test run with a previously saved model",
    )
    test_run.add_argument("model", type=Path, help="Path of previously saved model")

    full = sp.add_parser(
        "full-run", parents=[shared], help="Do a run with all the data"
    )
    full.add_argument("test_data", type=Path, help="Input files directory")
    full.add_argument(
        "-teb",
        "--test-eval-bitmaps",
        default="eval_bitmaps.npy",
        help="Override eval input bitmap filename",
    )
    full.add_argument(
        "-tea",
        "--test-eval-attributes",
        default="eval_attributes.npy",
        help="Override additional eval input attributes filename",
    )
    full.add_argument(
        "-tel",
        "--test-eval-labels",
        default="eval_labels.npy",
        help="Override eval input labels filename",
    )
    full.add_argument(
        "-tmb",
        "--test-mate-bitmaps",
        default="mate_bitmaps.npy",
        help="Override mate input bitmap filename",
    )
    full.add_argument(
        "-tma",
        "--test-mate-attributes",
        default="mate_attributes.npy",
        help="Override additional mate input attributes filename",
    )
    full.add_argument(
        "-tml",
        "--test-mate-labels",
        default="mate_labels.npy",
        help="Override mate input labels filename",
    )
    full.add_argument(
        "model", type=Path, help="Directory where created model will be saved"
    )
    full.add_argument(
        "plot", type=Path, help="Path and filename where history plot will be saved"
    )

    return parser.parse_args()


def load_and_slice_data(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eval_bitmaps = np.load(args.data / args.eval_bitmaps)
    n_eval = round(len(eval_bitmaps) * args.percentage / 100)
    eval_offset = round(len(eval_bitmaps) * args.offset / 100)

    mate_bitmaps = np.load(args.data / args.mate_bitmaps)
    n_mate = round(len(mate_bitmaps) * args.percentage / 100)
    mate_offset = round(len(mate_bitmaps) * args.offset / 100)

    bitmaps = np.concatenate(
        (
            eval_bitmaps[eval_offset: eval_offset + n_eval],
            mate_bitmaps[mate_offset: mate_offset + n_mate],
        )
    )

    attrs = np.concatenate(
        (
            np.load(args.data / args.eval_attributes)[eval_offset: eval_offset + n_eval],
            np.load(args.data / args.mate_attributes)[mate_offset: mate_offset + n_mate],
        )
    )

    labels = np.concatenate(
        (
            np.load(args.data / args.eval_labels)[eval_offset: eval_offset + n_eval],
            np.load(args.data / args.mate_labels)[mate_offset: mate_offset + n_mate],
        )
    )

    indices = np.arange(bitmaps.shape[0])
    np.random.shuffle(indices)

    return bitmaps[indices], attrs[indices], labels[indices]


def tune(args: argparse.Namespace) -> None:
    from model_parameter_pipeline import ModelParameterPipeline

    bitmaps, attributes, labels = load_and_slice_data(args)

    # Test parameter pipeline
    dict_of_params = {
        "batch_size": [512, 256],
        "dropout_rate": [0.3, 0.5],
        "epoch_number": [50],
    }

    model_param_pipeline = ModelParameterPipeline(
        bitmaps, attributes, labels, args.plots, args.models, dict_of_params
    )
    model_param_pipeline.run_pipeline()


def test(args: argparse.Namespace) -> None:
    from chess_evaluation_model import ChessEvaluationModel

    bitmaps, attributes, labels = load_and_slice_data(args)

    # TODO: Preprocess input before running model

    chess_eval = ChessEvaluationModel()
    chess_eval.load_model(args.model)
    chess_eval.test([bitmaps, attributes], labels)


def full_run(args: argparse.Namespace):
    from chess_evaluation_model import ChessEvaluationModel

    train_bitmaps = np.concatenate((np.load(args.data / args.eval_bitmaps),
                                    np.load(args.data / args.mate_bitmaps)))
    train_attributes = np.concatenate((np.load(args.data / args.eval_attributes),
                                       np.load(args.data / args.mate_attributes)))
    train_labels = np.concatenate((np.load(args.data / args.eval_labels),
                                   np.load(args.data / args.mate_labels)))
    test_bitmaps = np.concatenate((np.load(args.test_data / args.test_eval_bitmaps),
                                   np.load(args.test_data / args.test_mate_bitmaps)))
    test_attributes = np.concatenate((np.load(args.test_data / args.test_eval_attributes),
                                      np.load(args.test_data / args.test_mate_attributes)))
    test_labels = np.concatenate((np.load(args.test_data / args.test_eval_labels),
                                  np.load(args.test_data / args.test_mate_labels)))

    chess_eval = ChessEvaluationModel()
    chess_eval.initialize((8, 8, 12), (15,), "Adam", "relu", 0.3)

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

    if args.command == "pipeline":
        tune(args)
    elif args.command == "test":
        test(args)
    elif args.command == "full-run":
        full_run(args)


if __name__ == "__main__":
    main()
