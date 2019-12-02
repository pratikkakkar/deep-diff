from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

from .utils import save_animation


class Callback(object):
    """Callback base class.

    Attributes:
        model: instance of ``Model``. Reference of the model being trained.
    """

    def __init__(self):
        self.model = None

    def set_model(self, model):
        if model is not self.model:
            self.model = model
            self.init()

    def init(self):
        """Init after setting a model."""

    def on_epoch_begin(self):
        """Called at the beginning of every epoch."""

    def on_epoch_end(self):
        """Called at the end of every epoch."""

    def on_batch_begin(self):
        """Called at the beginning of every batch."""

    def on_batch_end(self):
        """Called at the end of every batch."""

    def on_train_begin(self):
        """Called at the beginning of model training."""

    def on_train_end(self):
        """Called at the end of model training."""

    def on_predict_begin(self):
        """Called at the beginning of prediction."""

    def on_predict_end(self):
        """Called at the end of prediction."""


class CallbackList(Callback):
    """Container abstracting a list of callbacks.

    Args:
        callbacks: List of ``Callback`` instances.
    """

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.model = None

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()

    def on_predict_begin(self):
        for callback in self.callbacks:
            callback.on_predict_begin()

    def on_predict_end(self):
        for callback in self.callbacks:
            callback.on_predict_end()

    def append(self, callback):
        if not isinstance(callback, Callback):
            raise Exception(str(callback) + " is an invalid Callback object")
        self.callbacks.append(callback)


class ModelCheckpoint(Callback):
    """Save the model after every epoch.

    Args:
        filepath (string): Path to save the model file.
        verbose: Verbosity mode, 0 or 1.
        save_better_only: If True, only save a better model according to the quantity monitored.
            Model is only checked at validation step according to ``display_every`` in ``Model.train``.
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, verbose=0, save_better_only=False, period=1):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.verbose = verbose
        self.save_better_only = save_better_only
        self.period = period

        self.monitor = "train loss"
        self.monitor_op = np.less
        self.epochs_since_last_save = 0
        self.best = np.Inf

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        if self.save_better_only:
            current = self.model.train_state.best_loss_train
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(
                        "Epoch {epoch}: {} improved from {:.2e} to {:.2e}, saving model to {}-{epoch} ...\n".format(
                            self.monitor,
                            self.best,
                            current,
                            self.filepath,
                            epoch=self.model.train_state.epoch,
                        )
                    )
                self.best = current
                self.model.save(self.filepath, verbose=0)
        else:
            self.model.save(self.filepath, verbose=self.verbose)


class EarlyStopping(Callback):
    """Stop training when a monitored quantity (training loss) has stopped improving.
    Only checked at validation step according to ``display_every`` in ``Model.train``.

    Args:
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
    """

    def __init__(self, min_delta=0, patience=0, baseline=None):
        super(EarlyStopping, self).__init__()

        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0

        self.monitor_op = np.less
        self.min_delta *= -1

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self):
        current = self.get_monitor_value()
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.model.train_state.epoch
                self.model.stop_training = True

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print("Epoch {}: early stopping".format(self.stopped_epoch))

    def get_monitor_value(self):
        return sum(self.model.train_state.loss_train)


class VariableValue(Callback):
    """Get the variable values.

    Args:
        var_list: A `TensorFlow Variable <https://www.tensorflow.org/api_docs/python/tf/Variable>`_
            or a list of TensorFlow Variable.
        period (int): Interval (number of epochs) between checking values.
        filename (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
            If ``None``, output to the screen.
    """

    def __init__(self, var_list, period=1, filename=None):
        super(VariableValue, self).__init__()
        self.var_list = var_list
        self.period = period

        self.file = sys.stdout if filename is None else open(filename, "w", buffering=1)
        self.value = None
        self.epochs_since_last = 0

    def on_train_begin(self):
        self.value = self.model.sess.run(self.var_list)
        print(self.model.train_state.epoch, self.value, file=self.file)

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.value = self.model.sess.run(self.var_list)
            print(self.model.train_state.epoch, self.value, file=self.file)

    def get_value(self):
        """Return the variable values."""
        return self.value


class OperatorPredictor(Callback):
    """Generates operator values for the input samples.

    Args:
        x: The input data.
        op: The operator with inputs (x, y).
    """

    def __init__(self, x, op):
        super(OperatorPredictor, self).__init__()
        self.x = x
        self.op = op
        self.tf_op = None
        self.value = None

    def init(self):
        self.tf_op = self.op(self.model.net.inputs, self.model.net.outputs)

    def on_predict_end(self):
        self.value = self.model.sess.run(
            self.tf_op,
            feed_dict=self.model._get_feed_dict(False, False, 2, self.x, None),
        )

    def get_value(self):
        return self.value


class FirstDerivative(OperatorPredictor):
    """Generates the first order derivative of the outputs with respect to the inputs.

    Args:
        x: The input data.
    """

    def __init__(self, x, component_x=0, component_y=0):
        def first_derivative(x, y):
            return tf.gradients(y[:, component_y : component_y + 1], x)[0][
                :, component_x : component_x + 1
            ]

        super(FirstDerivative, self).__init__(x, first_derivative)


class MovieDumper(Callback):
    """Dump a movie to show the training progress of the function along a line.

    Args:
        spectrum: If True, dump the spectrum of the Fourier transform.
    """

    def __init__(
        self,
        filename,
        x1,
        x2,
        num_points=100,
        period=1,
        component=0,
        save_spectrum=False,
        y_reference=None,
    ):
        super(MovieDumper, self).__init__()
        self.filename = filename
        x1 = np.array(x1)
        x2 = np.array(x2)
        self.x = x1 + (x2 - x1) / (num_points - 1) * np.arange(num_points)[:, None]
        self.period = period
        self.component = component
        self.save_spectrum = save_spectrum
        self.y_reference = y_reference

        self.y = []
        self.spectrum = []
        self.epochs_since_last_save = 0

    def init(self):
        self.tf_op = self.model.net.outputs[:, self.component]
        self.feed_dict = self.model._get_feed_dict(False, False, 2, self.x, None)

    def on_train_begin(self):
        self.y.append(self.model.sess.run(self.tf_op, feed_dict=self.feed_dict))
        if self.save_spectrum:
            A = np.fft.rfft(self.y[-1])
            self.spectrum.append(np.abs(A))

    def on_epoch_end(self):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            self.on_train_begin()

    def on_train_end(self):
        fname_x = self.filename + "_x.txt"
        fname_y = self.filename + "_y.txt"
        fname_movie = self.filename + "_y.gif"
        print(
            "\nSaving the movie of function to {}, {}, {}...".format(
                fname_x, fname_y, fname_movie
            )
        )
        np.savetxt(fname_x, self.x)
        np.savetxt(fname_y, np.array(self.y))
        if self.y_reference is None:
            save_animation(fname_movie, np.ravel(self.x), self.y)
        else:
            y_reference = np.ravel(self.y_reference(self.x))
            save_animation(
                fname_movie, np.ravel(self.x), self.y, y_reference=y_reference
            )

        if self.save_spectrum:
            fname_spec = self.filename + "_spectrum.txt"
            fname_movie = self.filename + "_spectrum.gif"
            print(
                "Saving the movie of spectrum to {}, {}...".format(
                    fname_spec, fname_movie
                )
            )
            np.savetxt(fname_spec, np.array(self.spectrum))
            xdata = np.arange(len(self.spectrum[0]))
            if self.y_reference is None:
                save_animation(fname_movie, xdata, self.spectrum, logy=True)
            else:
                A = np.fft.rfft(y_reference)
                save_animation(
                    fname_movie, xdata, self.spectrum, logy=True, y_reference=np.abs(A)
                )
