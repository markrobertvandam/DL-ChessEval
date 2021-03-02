import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Preprocess the chess dataset")
    parser.add_argument("-d", "--data")
    args = parser.parse_args()

    with open(args.data, 'rb') as file:
        files = np.load(file)
        bitmaps = files['bitmaps']
        print(bitmaps)


if __name__ == "__main__":
    main()
