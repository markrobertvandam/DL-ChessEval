import argparse
import csv
import os

import numpy as np


pieces = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11
}


def get_bitmap(board: str) -> np.ndarray:
    bitmap = np.zeros((8, 8, 12), np.uint8)

    rows = board.split("/")
    for row in range(8):
        col = 0
        idx = 0
        while col < 8:
            p = rows[row][idx]
            if p.isalpha():
                bitmap[row, col, pieces[p]] = 1
                col += 1
            else:
                col += int(p)
            idx += 1

    return bitmap


def get_attrs(fen_attrs: str):
    attrs = np.zeros(15, np.uint8)

    fen_attrs = fen_attrs.split(" ")

    # whose turn it is
    if fen_attrs[0] == 'b':
        attrs[0] = 1

    # who can castle and how
    if 'K' in fen_attrs[1]:
        attrs[1] = 1
    if 'Q' in fen_attrs[1]:
        attrs[2] = 1
    if 'k' in fen_attrs[1]:
        attrs[3] = 1
    if 'q' in fen_attrs[1]:
        attrs[4] = 1

    # square with legal en passant move
    if not fen_attrs[2] == '-':
        attrs[5 + ord(fen_attrs[2][0]) - 97] = 1

    attrs[13] = int(fen_attrs[3])
    attrs[14] = int(fen_attrs[4])

    return attrs


def get_eval(pos_eval: str) -> np.ndarray:
    if pos_eval[0] == "#":
        return np.array([1, int(pos_eval[1:])])
    else:
        return np.array([0, int(pos_eval)])


def preprocess(data_path: str, save_path: str) -> None:
    with open(data_path, 'r') as file:
        num_fens = sum(1 for _ in file)
        bitmaps = np.empty((num_fens, 8, 8, 12), np.uint8)
        attrs = np.empty((num_fens, 15), np.uint8)
        pos_evals = np.empty((num_fens, 2), np.int16)

    with open(data_path, 'r') as file:
        chess_reader = csv.reader(file)
        next(chess_reader)
        counter = 0
        for row in chess_reader:
            if not counter % 500000:
                print(f'\nRow {counter}', end='')
            elif not counter % 50000:
                print('.', end='')
            fen = row[0].split(" ", 1)
            bitmaps[counter] = get_bitmap(fen[0])
            attrs[counter] = get_attrs(fen[1])
            pos_evals[counter] = get_eval(row[1])

            counter += 1

        print(f'Total rows processed: {counter}')

        np.savez(
            os.path.join(save_path, "preprocessed_chess_dataset"),
            bitmaps=bitmaps,
            attrs=attrs,
            pos_evals=pos_evals
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    parser.add_argument("-d", "--data")
    parser.add_argument("-s", "--save")
    args = parser.parse_args()

    preprocess(args.data, args.save)


if __name__ == "__main__":
    main()
