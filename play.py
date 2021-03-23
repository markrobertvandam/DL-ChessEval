
from chess_evaluation_model import ChessEvaluationModel
from input_processing import ChessDataProcessor

import chess

def best_move(fen_position: str, fen_list: list) -> list:
    board = chess.Board(fen_position)
    print(fen_position)
    print(fen_list)
    best_move = ["", 0]
    for move in board.legal_moves:
        board.push(move)
        # analyze position and remember the best one, analyze is not yet implemented
        evaluation = analyze(board)
        if fen_list[1][0] == "w" and evaluation > best_move[1]:
            best_move = [move, evaluation]
        if fen_list[1][0] == "b" and evaluation < best_move[1]:
            best_move = [move, evaluation]
        board.pop() # undo last move
    return best_move

def kaufmann_test():
    with open('kaufmann.txt', 'r') as f:
        for line in f:
            line = line.split()
            [best_move, evaluation] = best_move(" ".join(line[:4]), [line[0], " ".join(line[1:4])])

kaufmann_test()