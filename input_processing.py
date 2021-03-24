from __future__ import annotations

import argparse
import csv
import os
from enum import Flag, auto
from functools import reduce
from typing import List

import numpy as np


class Fields(Flag):
    BITMAP = auto()
    ATTRIBUTE = auto()
    LABEL = auto()
    ALL = BITMAP | ATTRIBUTE | LABEL

    # def __str__(self):
    #     return self.name

    @staticmethod
    def from_string(s: str) -> Fields:
        try:
            return Fields[s.upper()]
        except KeyError:
            raise ValueError("Supply a valid Fields flag")


class ChessDataProcessor:
    piece_values = {
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
        "k": 11,
    }

    def __init__(self, data_path: str, save_dir: str) -> None:
        self.data_path = data_path
        self.save_dir = save_dir

    @staticmethod
    def __get_bitmap(board: str) -> np.ndarray:
        bitmap = np.zeros((8, 8, 12), np.bool)

        rows = board.split("/")
        for row in range(8):
            col = 0
            idx = 0
            while col < 8:
                p = rows[row][idx]
                if p.isalpha():
                    bitmap[row, col, ChessDataProcessor.piece_values[p]] = 1
                    col += 1
                else:
                    col += int(p)
                idx += 1

        return bitmap

    @staticmethod
    def __get_attrs(fen_attrs: str) -> np.ndarray:
        attrs = np.zeros(15, np.uint8)

        fen_attrs = fen_attrs.split(" ")

        # whose turn it is
        if fen_attrs[0] == "b":
            attrs[0] = 1

        # who can castle and how
        if "K" in fen_attrs[1]:
            attrs[1] = 1
        if "Q" in fen_attrs[1]:
            attrs[2] = 1
        if "k" in fen_attrs[1]:
            attrs[3] = 1
        if "q" in fen_attrs[1]:
            attrs[4] = 1

        # square with legal en passant move
        if not fen_attrs[2] == "-":
            attrs[5 + ord(fen_attrs[2][0]) - 97] = 1

        attrs[13] = int(fen_attrs[3])
        attrs[14] = int(fen_attrs[4])

        return attrs

    @staticmethod
    def __get_labels(pos_eval: str) -> np.ndarray:
        return (
            [0, int(pos_eval[1:]), 1] if pos_eval[0] == "#" else [int(pos_eval), 0, 0]
        )

    @staticmethod
    def __create_array(field: Fields, length: int) -> np.ndarray:
        if field is Fields.BITMAP:
            return np.empty((length, 8, 8, 12), np.bool)
        elif field is Fields.ATTRIBUTE:
            return np.empty((length, 15), np.uint8)
        elif field is Fields.LABEL:
            return np.empty(
                (length, 3),
                [("eval", np.int16), ("mate_turns", np.uint8), ("is_mate", np.bool)],
            )

        raise ValueError("field should be a singular Fields flag")

    @staticmethod
    def __get_value(field: Fields, row: List[str]):
        if field is Fields.BITMAP:
            return ChessDataProcessor.__get_bitmap(row[0].split(" ", 1)[0])
        elif field is Fields.ATTRIBUTE:
            return ChessDataProcessor.__get_attrs(row[0].split(" ", 1)[1])
        elif field is Fields.LABEL:
            return ChessDataProcessor.__get_labels(row[1])

        raise ValueError("field should be a singular Fields flag")

    def preprocess(self, fields: Fields) -> None:
        fields = [f for f in Fields if f in fields and f is not Fields.ALL]
        print(
            f"Preprocessing the following fields: {', '.join(f'{f.name.lower()}s' for f in fields)}"
        )

        print("Counting number of entries: ", end="")
        with open(self.data_path, "r") as file:
            data_length = sum(1 for _ in file)
        print(data_length)

        outputs = {f: self.__create_array(f, data_length) for f in fields}

        with open(self.data_path, "r") as file:
            chess_reader = csv.reader(file)
            next(chess_reader)
            counter = 0
            for row in chess_reader:
                if not counter % 500000:
                    print(f"\nRow {counter}", end="")
                elif not counter % 50000:
                    print(".", end="")
                for field in outputs:
                    outputs[field][counter] = self.__get_value(field, row)
                counter += 1

            print(f"\nTotal rows processed: {counter}\n")

        for field in outputs:
            print(f"Saving {field.name.lower()}s to {field.name.lower()}s.npy")
            np.save(
                os.path.join(self.save_dir, f"{field.name.lower()}s"), outputs[field]
            )

    def preprocess_fen_new(self, fen_string: str):
        fields = [f for f in Fields if f is not Fields.ALL]

        outputs = {f: self.__create_array(f, 1) for f in fields}

        for field in outputs:
            outputs[field][0] = self.__get_value(field, [fen_string, "0"])

        return outputs

    def preprocess_fen(self, fen_string: str):
        fen_board, fen_attrs = fen_string.split(" ", 1)
        return [
            np.array([self.__get_bitmap(fen_board)]),
            np.array([self.__get_attrs(fen_attrs)]),
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    parser.add_argument("--data", help="Path to the chess dataset")
    parser.add_argument("--save", help="Path to save the preprocessed data")
    parser.add_argument(
        "-f",
        "--fields",
        nargs="+",
        help="Fields to preprocess",
        type=Fields.from_string,
        choices=list(Fields),
        default=[Fields.ALL],
    )
    args = parser.parse_args()

    cdp = ChessDataProcessor(args.data, args.save)
    cdp.preprocess(reduce(Fields.__or__, args.fields))


if __name__ == "__main__":
    main()
