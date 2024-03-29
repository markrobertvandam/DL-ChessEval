from chess_evaluation_model import ChessEvaluationModel
from input_processing import ChessDataProcessor

import argparse
import chess


class PlayChess:

    def __init__(self):
        self.chess_eval = ChessEvaluationModel()

    def analyze(self, *args):
        if len(args) == 0:
            input1 = input("Paste fen-string: ")
            curr_board = chess.Board(" ".join(input1.split()[:4]))
        else:
            curr_board = args[0]

        fen_string = curr_board.fen()
        bitmap, attr = ChessDataProcessor.preprocess_fen(fen_string)
        prediction = self.chess_eval.predict([bitmap, attr])
        if len(args) == 0:
            print(prediction)
        return prediction[0]

    def predict_best_move(self, curr_board: chess.Board) -> list:
        turn = curr_board.turn
        eval_moves = []
        mate_moves = []
        opponent_mates = []

        for move in curr_board.legal_moves:
            new_board = curr_board.copy()
            new_board.push(move)
            # analyze position and remember the best one, analyze is not yet implemented
            prediction = self.analyze(new_board)
            evaluation = prediction[0].item()
            moves_til_mate = prediction[1].item()
            is_mate = prediction[2].item()

            if turn:
                if is_mate > 0.5 and moves_til_mate >= 0:
                    mate_moves.append((curr_board.san(move), moves_til_mate))
                elif is_mate > 0.5:
                    opponent_mates.append((curr_board.san(move), moves_til_mate))
                else:
                    eval_moves.append((curr_board.san(move), evaluation))
            else:
                if is_mate > 0.5 and moves_til_mate <= 0:
                    mate_moves.append((curr_board.san(move), moves_til_mate))
                elif is_mate > 0.5:
                    opponent_mates.append((curr_board.san(move), moves_til_mate))
                else:
                    eval_moves.append((curr_board.san(move), evaluation))

        if turn:
            eval_moves.sort(key=lambda x: x[1], reverse=True)
            mate_moves.sort(key=lambda x: x[1])
            opponent_mates.sort(key=lambda x: x[1])
        else:
            eval_moves.sort(key=lambda x: x[1])
            mate_moves.sort(key=lambda x: x[1], reverse=True)
            opponent_mates.sort(key=lambda x: x[1], reverse=True)

        return eval_moves, mate_moves, opponent_mates

    def kaufman_test(self):

        with open("kaufman.txt", "r") as f:
            for line in f:
                predict_true = line.split(";")[0].split()[-1]
                line = line.split()
                board = chess.Board(" ".join(line[:4]))
                print(self.predict_look_ahead(board), predict_true)

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
        stale_mates = []
        eval_moves, mate_moves, opponent_mates = self.predict_best_move(board)

        if len(eval_moves) > 0 or len(mate_moves) > 0:
            best_moves = set(eval_moves[:12] + mate_moves)
        else:
            best_moves = opponent_mates
        for move in best_moves:
            opp_eval_moves, opp_mate_moves, opp_opponent_mates = [], [], []
            new_board = board.copy()
            new_board.push_san(move[0])

            if new_board.is_checkmate():
                if len(mate_moves) > 0:
                    potential_mate_moves.append(move)
                else:
                    potential_eval_moves.append(move)
            elif new_board.is_stalemate():
                stale_mates.append(move)
            else:
                opp_eval_moves, opp_mate_moves, opp_opponent_mates = self.predict_best_move(new_board)

                if len(opp_opponent_mates) > 0:
                    potential_mate_moves.append((move[0], opp_opponent_mates[0][1]))
                elif len(opp_eval_moves) > 0:
                    potential_eval_moves.append((move[0], opp_eval_moves[0][1]))
                else:
                    potential_opponent_mates.append((move[0], opp_mate_moves[0][1]))

        if turn:
            potential_mate_moves.sort(key=lambda x: x[1])
            potential_opponent_mates.sort(key=lambda x: x[1])
            potential_eval_moves.sort(key=lambda x: x[1], reverse=True)
        else:
            potential_mate_moves.sort(key=lambda x: x[1], reverse=True)
            potential_opponent_mates.sort(key=lambda x: x[1], reverse=True)
            potential_eval_moves.sort(key=lambda x: x[1])

        if len(potential_mate_moves) > 0:
            return potential_mate_moves[0]
        if len(potential_eval_moves) > 0:
            return potential_eval_moves[0]
        if len(stale_mates) > 0:
            return stale_mates[0]
        return opponent_mates[0]

    def play_game(self, colour: int):
        board = chess.Board()
        turn = 1
        while not (board.is_checkmate()):
            if colour == turn:
                while True:
                    input1 = input("Please give your move: ")
                    try:
                        board.push_san(input1)
                        colour *= -1
                        break
                    except ValueError:
                        print("Invalid move, try again")
            else:
                model_move = self.predict_look_ahead(board)
                print(model_move)
                board.push_san(model_move[0])
                colour *= -1
        print("end of game")


def main():
    play_chess = PlayChess()
    commands = {'kaufman': play_chess.kaufman_test,
                'predict': play_chess.predict_fen,
                'predict_look_ahead': play_chess.predict_look_ahead,
                'play': play_chess.play_game,
                'analyze': play_chess.analyze}

    parser = argparse.ArgumentParser(description="Run Kaufman test or play chess")
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
