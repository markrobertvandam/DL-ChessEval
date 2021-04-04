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
        return (
            (0, int(pos_eval[1:]), 1) if pos_eval[0] == "#" else (int(pos_eval), 0, 0)
        )

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
        fields = [
            f for f in ChessDataFields if f in fields and f is not ChessDataFields.ALL
        ]
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
            print(
                f"Saving eval {field.name.lower()}s to eval_{field.name.lower()}s.npy"
            )
            np.save(self.save_dir / f"eval_{field.name.lower()}s", eval_outputs[field])
            print(
                f"Saving mate {field.name.lower()}s to mate_{field.name.lower()}s.npy"
            )
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

    @staticmethod
    def split_train_test(data_path, train_percentage, train_path, test_path):
        eval_bitmaps = np.load(os.path.join(data_path, "eval_bitmaps.npy"))
        eval_attributes = np.load(os.path.join(data_path, "eval_attributes.npy"))
        eval_labels = np.load(os.path.join(data_path, "eval_labels.npy"))

        mate_bitmaps = np.load(os.path.join(data_path, "mate_bitmaps.npy"))
        mate_attributes = np.load(os.path.join(data_path, "mate_attributes.npy"))
        mate_labels = np.load(os.path.join(data_path, "mate_labels.npy"))

        index_splitter_eval = int(len(eval_bitmaps) * train_percentage)
        index_splitter_mate = int(len(mate_bitmaps) * train_percentage)

        # Train eval
        train_eval_bitmaps = eval_bitmaps[:index_splitter_eval]
        train_eval_attributes = eval_attributes[:index_splitter_eval]
        train_eval_labels = eval_labels[:index_splitter_eval]
        # Test eval
        test_eval_bitmaps = eval_bitmaps[index_splitter_eval:]
        test_eval_attributes = eval_attributes[index_splitter_eval:]
        test_eval_labels = eval_labels[index_splitter_eval:]
        # Train mate
        train_mate_bitmaps = mate_bitmaps[:index_splitter_mate]
        train_mate_attributes = mate_attributes[:index_splitter_mate]
        train_mate_labels = mate_labels[:index_splitter_mate]
        # Test mate
        test_mate_bitmaps = mate_bitmaps[index_splitter_mate:]
        test_mate_attributes = mate_attributes[index_splitter_mate:]
        test_mate_labels = mate_labels[index_splitter_mate:]

        # Save train eval
        np.save(
            os.path.join(
                train_path, "train_eval_bitmaps_{}.npy".format(train_percentage)
            ),
            train_eval_bitmaps,
        )
        np.save(
            os.path.join(
                train_path, "train_eval_attributes_{}.npy".format(train_percentage)
            ),
            train_eval_attributes,
        )
        np.save(
            os.path.join(
                train_path, "train_eval_labels_{}.npy".format(train_percentage)
            ),
            train_eval_labels,
        )
        # Save test eval
        np.save(
            os.path.join(
                test_path,
                "test_eval_bitmaps_{}.npy".format(round(1 - train_percentage, 2)),
            ),
            test_eval_bitmaps,
        )
        np.save(
            os.path.join(
                test_path,
                "test_eval_attributes_{}.npy".format(round(1 - train_percentage, 2)),
            ),
            test_eval_attributes,
        )
        np.save(
            os.path.join(
                test_path,
                "test_eval_labels_{}.npy".format(round(1 - train_percentage, 2)),
            ),
            test_eval_labels,
        )
        # Save train mate
        np.save(
            os.path.join(
                train_path, "train_mate_bitmaps_{}.npy".format(train_percentage)
            ),
            train_mate_bitmaps,
        )
        np.save(
            os.path.join(
                train_path, "train_mate_attributes_{}.npy".format(train_percentage)
            ),
            train_mate_attributes,
        )
        np.save(
            os.path.join(
                train_path, "train_mate_labels_{}.npy".format(train_percentage)
            ),
            train_mate_labels,
        )
        # Save test mate
        np.save(
            os.path.join(
                test_path,
                "test_mate_bitmaps_{}.npy".format(round(1 - train_percentage, 2)),
            ),
            test_eval_bitmaps,
        )
        np.save(
            os.path.join(
                test_path,
                "test_mate_attributes_{}.npy".format(round(1 - train_percentage, 2)),
            ),
            test_eval_attributes,
        )
        np.save(
            os.path.join(
                test_path,
                "test_mate_labels_{}.npy".format(round(1 - train_percentage, 2)),
            ),
            test_eval_labels,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")

    sp = parser.add_subparsers(title="Command", dest="command")

    split_process = sp.add_parser(
        "split-data", help="Split train test data into separate files"
    )
    split_process.add_argument(
        "saved_preprocessed_data",
        help="Path to the already saved preprocessed data using 'process-input' command",
    )
    split_process.add_argument(
        "train_percentage",
        type=float,
        default=0.9,
        help="Set percantage to split train test",
    )
    split_process.add_argument(
        "train_save", type=Path, help="Directory where train data will be saved"
    )
    split_process.add_argument(
        "test_save", type=Path, help="Directory where test data will be saved"
    )

    process_input = sp.add_parser("process-input", help="Process the raw chess dataset")
    process_input.add_argument("data", type=Path, help="Path to the chess dataset")
    process_input.add_argument(
        "save", type=Path, help="Directory where preprocessed data will be saved"
    )
    process_input.add_argument(
        "-f",
        "--fields",
        nargs="+",
        help="Fields to preprocess",
        type=ChessDataFields.from_string,
        choices=list(ChessDataFields),
        default=[ChessDataFields.ALL],
    )

    args = parser.parse_args()

    if args.command == "process-input":
        cdp = ChessDataProcessor(args.data, args.save)
        cdp.preprocess(reduce(ChessDataFields.__or__, args.fields))
    elif args.command == "split-data":
        ChessDataProcessor.split_train_test(
            args.saved_preprocessed_data,
            args.train_percentage,
            args.train_save,
            args.test_save,
        )


if __name__ == "__main__":
    main()
