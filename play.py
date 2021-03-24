from chess_evaluation_model import ChessEvaluationModel
from input_processing import ChessDataProcessor

import chess

input_processing_obj = ChessDataProcessor("", "")


def analyze(curr_board):
    fen_string = curr_board.fen()
    res = input_processing_obj.preprocess_fen(fen_string)
    curr_bitmap = res['BITMAP']
    curr_attributes = res['ATTRIBUTES']
    
    print(0)

def predict_best_move(fen_position: str) -> list:
    board = chess.Board(fen_position)
    print(fen_position)
    best_move = ["", 0]
    for move in board.legal_moves:
        board.push(move)
        # analyze position and remember the best one, analyze is not yet implemented
        evaluation = analyze(board)
        if fen_list[1][0] == "w" and evaluation > best_move[1]:
            best_move = [move, evaluation]
        if fen_list[1][0] == "b" and evaluation < best_move[1]:
            best_move = [move, evaluation]
        board.pop()  # undo last move
    return best_move


def kaufmann_test():
    with open("kaufmann.txt", "r") as f:
        for line in f:
            line = line.split()
            result_best_move = predict_best_move(" ".join(line[:4]))


kaufmann_test()