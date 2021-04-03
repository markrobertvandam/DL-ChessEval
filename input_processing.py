from __future__ import annotations

import argparse
import csv
import os
from enum import Flag, auto
from functools import reduce
from pathlib import Path
from typing import List, Tuple

import numpy as np


class ChessDataFields(Flag):
    BITMAP = auto()
    ATTRIBUTE = auto()
    LABEL = auto()
    ALL = BITMAP | ATTRIBUTE | LABEL

    @staticmethod
    def from_string(s: str) -> ChessDataFields:
        try:
            return ChessDataFields[s.upper()]
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

    def __init__(self, data_path: Path, save_dir: Path) -> None:
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
    def __get_labels(pos_eval: str) -> Tuple[int, int, int]:
        return (0, int(pos_eval[1:]), 1) if pos_eval[0] == "#" else (int(pos_eval), 0, 0)

    @staticmethod
    def __create_array(field: ChessDataFields, length: int) -> np.ndarray:
        if field is ChessDataFields.BITMAP:
            return np.empty((length, 8, 8, 12), np.bool)
        elif field is ChessDataFields.ATTRIBUTE:
            return np.empty((length, 15), np.uint8)
        elif field is ChessDataFields.LABEL:
            return np.empty(
                (length,),
                [("eval", np.int16), ("mate_turns", np.int16), ("is_mate", np.bool)],
            )

        raise ValueError("field should be a singular Fields flag")

    @staticmethod
    def __get_value(field: ChessDataFields, row: List[str]):
        if field is ChessDataFields.BITMAP:
            return ChessDataProcessor.__get_bitmap(row[0].split(" ", 1)[0])
        elif field is ChessDataFields.ATTRIBUTE:
            return ChessDataProcessor.__get_attrs(row[0].split(" ", 1)[1])
        elif field is ChessDataFields.LABEL:
            return ChessDataProcessor.__get_labels(row[1])
        raise ValueError("field should be a singular Fields flag")

    def preprocess(self, fields: ChessDataFields) -> None:
        fields = [f for f in ChessDataFields if f in fields and f is not ChessDataFields.ALL]
        print(
            f"Preprocessing the following fields: {', '.join(f'{f.name.lower()}s' for f in fields)}"
        )

        print("Counting number of entries: ", end="")
        num_eval = 0
        num_mate = 0
        with open(self.data_path, "r") as file:
            # Skip header before counting lines
            next(file)
            for line in file:
                if "#" in line:
                    num_mate += 1
                else:
                    num_eval += 1
        print(f"eval: {num_eval}, mate: {num_mate}")

        eval_outputs = {f: self.__create_array(f, num_eval) for f in fields}
        mate_outputs = {f: self.__create_array(f, num_mate) for f in fields}

        with open(self.data_path, "r") as file:
            chess_reader = csv.reader(file)
            next(chess_reader)
            mate_counter = 0
            eval_counter = 0
            for row in chess_reader:
                if not (mate_counter + eval_counter) % 500000:
                    print(f"\nRow {mate_counter + eval_counter}", end="", flush=True)
                elif not (mate_counter + eval_counter) % 50000:
                    print(".", end="", flush=True)
                # Select correct output dict
                if "#" in row[1]:
                    for field in fields:
                        mate_outputs[field][mate_counter] = self.__get_value(field, row)
                    mate_counter += 1
                else:
                    for field in fields:
                        eval_outputs[field][eval_counter] = self.__get_value(field, row)
                    eval_counter += 1

            print(f"\nTotal rows processed: {mate_counter + eval_counter}\n")

        for field in fields:
            print(f"Saving eval {field.name.lower()}s to eval_{field.name.lower()}s.npy")
            np.save(self.save_dir / f"eval_{field.name.lower()}s", eval_outputs[field])
            print(f"Saving mate {field.name.lower()}s to mate_{field.name.lower()}s.npy")
            np.save(self.save_dir / f"mate_{field.name.lower()}s", mate_outputs[field])

    @staticmethod
    def preprocess_fen_new(fen_string: str):
        fields = [f for f in ChessDataFields if f is not ChessDataFields.ALL]

        outputs = {f: ChessDataProcessor.__create_array(f, 1) for f in fields}

        for field in outputs:
            outputs[field][0] = ChessDataProcessor.__get_value(field, [fen_string, "0"])

        return outputs

    @staticmethod
    def preprocess_fen(fen_string: str):
        fen_board, fen_attrs = fen_string.split(" ", 1)
        return [
            np.array([ChessDataProcessor.__get_bitmap(fen_board)]),
            np.array([ChessDataProcessor.__get_attrs(fen_attrs)]),
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    parser.add_argument("data", type=Path, help="Path to the chess dataset")
    parser.add_argument("save", type=Path, help="Directory where preprocessed data will be saved")
    parser.add_argument(
        "-f",
        "--fields",
        nargs="+",
        help="Fields to preprocess",
        type=ChessDataFields.from_string,
        choices=list(ChessDataFields),
        default=[ChessDataFields.ALL],
    )
    args = parser.parse_args()

    cdp = ChessDataProcessor(args.data, args.save)
    cdp.preprocess(reduce(ChessDataFields.__or__, args.fields))


if __name__ == "__main__":
    main()
