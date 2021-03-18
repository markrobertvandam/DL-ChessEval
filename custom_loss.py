import tensorflow as tf
from tensorflow.python.keras.losses import BinaryCrossentropy, MeanSquaredError


def custom_loss(y_true, y_pred):
    print(y_true)
    print(y_pred)
    exit()
    # Apparently y_true is a tensor of shape (None, 1) and I don't know how to unpack the original bits
    (eval_pred, mate_pred, is_mate_pred) = tf.split(y_pred, num_or_size_splits=3)
    (eval_true, mate_true, is_mate_true) = tf.split(y_true, num_or_size_splits=3)
    bce = BinaryCrossentropy()
    mse = MeanSquaredError()
    loss_is_mate = bce(is_mate_true, is_mate_pred)
    loss_eval = mse(eval_true, eval_pred)
    loss_mate = mse(mate_true, eval_true)
    print(f"is_mate_loss: {loss_is_mate.print_tensor()} - eval_loss: {loss_eval} - mate_loss: {loss_mate}")
    return loss_is_mate + 10 * loss_eval + loss_mate
