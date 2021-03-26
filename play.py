from chess_evaluation_model import ChessEvaluationModel
from input_processing import ChessDataProcessor

import argparse
import chess


class PlayChess:

    def __init__(self):
        self.input_processing_obj = ChessDataProcessor("","")
        self.chess_eval = ChessEvaluationModel()

    def analyze(self, curr_board):
        fen_string = curr_board.fen()
        bitmap, attr = self.input_processing_obj.preprocess_fen(fen_string)
        prediction = self.chess_eval.predict([bitmap, attr])

        return prediction

    def predict_best_move(self, curr_board) -> list:
        turn = curr_board.turn
        eval_moves = []
        mate_moves = []
        opponent_mates = []

        for move in curr_board.legal_moves:
            new_board = curr_board.copy()
            new_board.push(move)
            # analyze position and remember the best one, analyze is not yet implemented
            prediction = self.analyze(new_board)

            eval = prediction[0].item()
            moves_till_mate = prediction[1].item()
            is_mate = prediction[2].item()

            if turn:
                if is_mate > 0.5 and moves_till_mate >= 0:
                    mate_moves.append((curr_board.san(move), moves_till_mate))
                elif is_mate > 0.5:
                    opponent_mates.append((curr_board.san(move), moves_till_mate))
                else:
                    eval_moves.append((curr_board.san(move), eval))
            else:
                if is_mate > 0.5 and moves_till_mate <= 0:
                    mate_moves.append((curr_board.san(move), moves_till_mate))
                elif is_mate > 0.5:
                    opponent_mates.append((curr_board.san(move), moves_till_mate))
                else:
                    eval_moves.append((curr_board.san(move), eval))

        if turn:
            eval_moves.sort(key=lambda x: x[1], reverse = True)
            mate_moves.sort(key=lambda x: x[1])
            opponent_mates.sort(key=lambda x: x[1])
        else:
            eval_moves.sort(key=lambda x: x[1])
            mate_moves.sort(key=lambda x: x[1], reverse = True)
            opponent_mates.sort(key = lambda x: x[1], reverse = True)

        return eval_moves, mate_moves, opponent_mates

    def kaufmann_test(self):

        with open("kaufmann.txt", "r") as f:
            for line in f:
                line = line.split()
                board = chess.Board(" ".join(line[:4]))
                eval_moves, mate_moves, opponent_mates = self.predict_best_move(board)
                print(self.predict_look_ahead(board))
                print(eval_moves)
                print(mate_moves)
                print(opponent_mates)

    def predict_fen(self, *args) -> str:
        if len(args) == 0:
            input1 = input("Paste fen-string: ")
            board = chess.Board(" ".join(input1.split()[:4]))
        else:
            board = args[0]

        eval_moves, mate_moves, opponent_mates = self.predict_best_move(board)
        print(eval_moves, mate_moves, opponent_mates)
        if len(mate_moves) > 0:
            return mate_moves[0]
        if len(eval_moves) > 0:
            return eval_moves[0]
        return opponent_mates[0]

    def predict_look_ahead(self, *args):
        if len(args) == 0:
            input1 = input("Paste fen-string: ")
            board = chess.Board(" ".join(input1.split()[:4]))
        else:
            board = args[0]
        turn = board.turn
        potential_mate_moves = []
        potential_eval_moves = []
        potential_opponent_mates = []
        eval_moves, opponent_mates, mate_moves = self.predict_best_move(board)

        if len(eval_moves) > 0 or len(mate_moves) > 0:
            best_moves = set(eval_moves[:6] + mate_moves)
        else:
            best_moves = opponent_mates
        for move in best_moves:
            new_board = board.copy()
            new_board.push_san(move[0])
            eval_moves, opponent_mates, mate_moves = self.predict_best_move(new_board)
            if len(mate_moves) > 0:
                potential_mate_moves.append((move[0], mate_moves[0][1]))
            elif len(eval_moves) > 0:
                potential_eval_moves.append((move[0], eval_moves[0][1]))
            else:
                potential_opponent_mates.append((move[0], opponent_mates[0][1]))
        if turn:
            potential_mate_moves.sort(key = lambda x: x[1])
            potential_opponent_mates.sort(key = lambda x: x[1])
            potential_eval_moves.sort(key = lambda x: x[1], reverse = True)
        else:
            potential_mate_moves.sort(key = lambda x: x[1], reverse = True)
            potential_opponent_mates.sort(key = lambda x: x[1], reverse = True)
            potential_eval_moves.sort(key = lambda x: x[1])

        print(potential_mate_moves, potential_eval_moves, potential_opponent_mates)
        if len(potential_mate_moves) > 0:
            return potential_mate_moves[0]
        if len(potential_eval_moves) > 0:
            return potential_eval_moves[0]
        return opponent_mates[0]

    def play_game(self, colour: int):
        board = chess.Board()
        turn = 1
        while not(board.is_checkmate()):
            if colour == turn:
                input1 = input("Please give your move: ")
                board.push_san(input1)
                colour *= -1
            else:
                model_move = self.predict_fen(board)
                print(model_move)
                board.push_san(model_move[0])
                colour *= -1
        print("end of game")

def main():
    play_chess = PlayChess()
    commands = {'kaufmann': play_chess.kaufmann_test,
                'predict': play_chess.predict_fen,
                'predict_look_ahead': play_chess.predict_look_ahead,
                'play': play_chess.play_game}

    parser = argparse.ArgumentParser(description="Run Kaufmann test or play chess")
    parser.add_argument(
        "-m", "--model", help="Path to the model", required=True
    )
    parser.add_argument(
        "-c", "--command", choices=commands.keys(), help="Parameter to choose function", required=True
    )

    args = parser.parse_args()
    func = commands[args.command]
    play_chess.chess_eval.load_model(args.model)
    func()


if __name__ == "__main__":
    main()
