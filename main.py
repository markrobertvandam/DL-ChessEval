import argparse

import numpy as np

from chess_evaluation_model import ChessEvaluationModel


def main():
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    parser.add_argument("-d", "--data")
    args = parser.parse_args()

    files = np.load(args.data)
    bitmaps = files['bitmaps']
    n_samples = len(bitmaps) * 0.05
    bitmaps = bitmaps[:n_samples]
    attrs = files['attrs'][:n_samples]
    pos_evals = files['pos_evals'][:n_samples]

    chess_eval_model = ChessEvaluationModel((8, 8, 12), (15,))

    # chess_eval_model.compile()


if __name__ == "__main__":
    main()
