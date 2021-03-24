from chess_evaluation_model import ChessEvaluationModel
from input_processing import ChessDataProcessor

import argparse
import chess

input_processing_obj = ChessDataProcessor("", "")
chess_eval = ChessEvaluationModel()


def analyze(curr_board):
    fen_string = curr_board.fen()
    bitmap, attr = input_processing_obj.preprocess_fen(fen_string)
    prediction = chess_eval.predict([bitmap, attr])
    
    return prediction

def predict_best_move(fen_position: str) -> list:
    board = chess.Board(fen_position)
    print(fen_position)
    turn = board.turn

    eval_moves = []
    mate_moves = []
    for move in board.legal_moves:
        new_board = board.copy()
        new_board.push(move)
        # analyze position and remember the best one, analyze is not yet implemented
        prediction = analyze(new_board)


        eval = prediction[0].item()
        moves_till_mate = prediction[1].item()
        is_mate = prediction[2].item()

        if turn:
            if is_mate > 0.5 and moves_till_mate >= 0:
                mate_moves.append((board.san(move), moves_till_mate))
            else:
                eval_moves.append((board.san(move), eval))
        else:
            if is_mate > 0.5 and moves_till_mate <= 0:
                mate_moves.append((board.san(move), moves_till_mate))
            else:
                eval_moves.append((board.san(move), eval))

    if turn:
        eval_moves.sort(key=lambda x: x[1], reverse = True)
        mate_moves.sort(key=lambda x: x[1])
    else:
        eval_moves.sort(key=lambda x: x[1])
        mate_moves.sort(key=lambda x: x[1], reverse = True)

    print(eval_moves)
    print(mate_moves)
    return "hoi"


def kaufmann_test():
    parser = argparse.ArgumentParser(description = "Run Kaufmann test")
    parser.add_argument(
        "-m", "--model", help = "Path to the model", required = True
    )
    args = parser.parse_args()
    chess_eval.load_model(args.model)
    with open("kaufmann.txt", "r") as f:
        for line in f:
            line = line.split()
            result_best_move = predict_best_move(" ".join(line[:4]))


kaufmann_test()