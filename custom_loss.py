import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.python.keras.metrics import binary_accuracy


class CustomLossMetrics:
    def __init__(self):
        self.bce = BinaryCrossentropy()
        self.mse = MeanSquaredError()

    def overall_loss(self, y_true, y_pred):
        (eval_true, mate_true, is_mate_true) = tf.split(
            y_true, num_or_size_splits=3, axis=1
        )
        (eval_pred, mate_pred, is_mate_pred) = tf.split(
            y_pred, num_or_size_splits=3, axis=1
        )
        eval_pred = tf.where(tf.equal(is_mate_true, 0), eval_pred, eval_true)
        mate_pred = tf.where(tf.equal(is_mate_true, 1), mate_pred, mate_true)
        eval_loss = self.mse(eval_true, eval_pred)
        mate_loss = self.mse(mate_true, mate_pred)
        is_mate_loss = self.bce(is_mate_true, is_mate_pred)
        # print(f"is_mate_loss: {loss_is_mate} - eval_loss: {loss_eval} - mate_loss: {loss_mate}")
        # tf.print(" - is_mate_loss: ", loss_is_mate, "- eval_loss: ", loss_eval, "- mate_loss:", loss_mate, end="\r")
        return 2e3 * eval_loss + mate_loss + 10 * is_mate_loss

    def eval_mse(self, y_true, y_pred):
        # Apparently y_true is a tensor of shape (None, 1) and I don't know how to unpack the original bits
        (eval_true, _, is_mate_true) = tf.split(y_true, num_or_size_splits=3, axis=1)
        (eval_pred, _, _) = tf.split(y_pred, num_or_size_splits=3, axis=1)
        eval_pred = tf.where(tf.equal(is_mate_true, 0), eval_pred, eval_true)
        return self.mse(eval_true, eval_pred)

    def mate_mse(self, y_true, y_pred):
        # Apparently y_true is a tensor of shape (None, 1) and I don't know how to unpack the original bits
        (_, mate_true, is_mate_true) = tf.split(y_true, num_or_size_splits=3, axis=1)
        (_, mate_pred, _) = tf.split(y_pred, num_or_size_splits=3, axis=1)
        mate_pred = tf.where(tf.equal(is_mate_true, 1), mate_pred, mate_true)
        return self.mse(mate_true, mate_pred)

    @staticmethod
    def is_mate_ba(y_true, y_pred):
        # Apparently y_true is a tensor of shape (None, 1) and I don't know how to unpack the original bits
        (_, _, is_mate_true) = tf.split(y_true, num_or_size_splits=3, axis=1)
        (_, _, is_mate_pred) = tf.split(y_pred, num_or_size_splits=3, axis=1)
        return binary_accuracy(is_mate_true, is_mate_pred)
