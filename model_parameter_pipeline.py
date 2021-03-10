import tensorflow as tf
from chess_evaluation_model import ChessEvaluationModel


class ModelParameterPipeline:
    def __init__(
        self, bitmaps, attributes, target_eval, target_mate, target_is_mate, plot_path
    ):
        self.bitmaps = bitmaps
        self.attributes = attributes
        self.target_eval = target_eval
        self.target_mate = target_mate
        self.target_is_mate = target_is_mate
        self.plot_path = plot_path
