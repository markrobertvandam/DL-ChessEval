import argparse

import numpy as np

from chess_evaluation_model import ChessEvaluationModel


def main():
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    parser.add_argument("bitmaps", help="Path to the input bitmaps")
    parser.add_argument("attributes", help="Path to the additional input attributes")
    parser.add_argument("labels", help="Path to the input labels")
    parser.add_argument("plot", help="Path to the generated history plot image")
    args = parser.parse_args()

    bitmaps = np.load(args.bitmaps)
    n_samples = round(len(bitmaps) * 0.05)
    bitmaps = bitmaps[:n_samples]
    attributes = np.load(args.attributes)[:n_samples]
    labels = np.load(args.labels)[:n_samples]

    chess_eval = ChessEvaluationModel()
    chess_eval.initialize((8, 8, 12), (15,))
    history = chess_eval.train(
        [bitmaps, attributes],
        [labels['eval'], labels['mate_turns'], labels['is_mate']],
        10,
        256
    )
    chess_eval.plot_history(history, args.plot)


if __name__ == "__main__":
    main()
