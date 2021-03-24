from chess_evaluation_model import ChessEvaluationModel
from input_processing import ChessDataProcessor

import chess

input_processing_obj = ChessDataProcessor("", "")
chess_eval = ChessEvaluationModel()
chess_eval.load_model("D:\\PythonProjects\\DL-ChessEval\\data\\models\\af_relu_op_Adam_dr_0.3_bs_512_ep_25_lr_0.01")

def analyze(curr_board):
    fen_string = curr_board.fen()
    bitmap, attr = input_processing_obj.preprocess_fen(fen_string)
    prediction = chess_eval.predict([bitmap, attr])
    
    return prediction

def predict_best_move(fen_position: str) -> list:
    board = chess.Board(fen_position)
    print(fen_position)
    best_move = ["", 0]
    turn = board.turn
    for move in board.legal_moves:
        board.push(move)
        # analyze position and remember the best one, analyze is not yet implemented
        prediction = analyze(board)
        
        is_mate = prediction[2].item()
        if is_mate > 0.5:
            pass
        else:
            evaluation = prediction[0].item()
            
            if turn and evaluation > best_move[1]:
                best_move = [move, evaluation]
            if not turn and evaluation < best_move[1]:
                best_move = [move, evaluation]
        
        board.pop()  # undo last move
    return best_move


def kaufmann_test():
    with open("kaufmann.txt", "r") as f:
        for line in f:
            line = line.split()
            result_best_move = predict_best_move(" ".join(line[:4]))


kaufmann_test()