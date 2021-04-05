"""
Custom early stopping callback to combine eval and mate loss into a single early stopping criterion.

Code used was taken and adapted from:
https://stackoverflow.com/questions/64556120/early-stopping-with-multiple-conditions
"""

import numpy as np
from tensorflow.python.keras.callbacks import Callback


class CustomEarlyStopping(Callback):
    def __init__(self, patience=10, d_eval=0.005, d_mate=0.005) -> None:
        super(CustomEarlyStopping, self).__init__()
        self.patience = patience
        self.d_eval = d_eval
        self.d_mate = d_mate
        self.best_weights = None
        self.wait = None
        self.stopped_epoch = None
        self.best_eval_loss = None
        self.best_mate_loss = None

    def on_train_begin(self, logs=None) -> None:
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_eval_loss = np.Inf
        self.best_mate_loss = np.Inf

    def on_epoch_end(self, epoch, logs=None) -> None:
        eval_loss = logs.get("val_eval_loss")
        mate_loss = logs.get("val_mate_loss")

        should_stop = True

        if np.less_equal(eval_loss, self.best_eval_loss - self.d_eval):
            self.best_eval_loss = eval_loss
            should_stop = False

        if np.less_equal(mate_loss, self.best_mate_loss - self.d_mate):
            self.best_mate_loss = mate_loss
            should_stop = False

        if not should_stop:
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            return

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            print("Restoring model weights from the end of the best epoch.")
            self.model.set_weights(self.best_weights)
