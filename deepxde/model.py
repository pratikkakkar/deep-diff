from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from . import display
from . import losses as losses_module
from . import metrics as metrics_module
from . import train as train_module
from .callbacks import CallbackList
from .utils import guarantee_initialized_variables, timing


class Model(object):
    """The ``Model`` class trains a ``Map`` on a ``Data``.

    Args:
        data: ``deepxde.data.Data`` instance.
        net: ``deepxde.maps.Map`` instance.
    """

    def __init__(self, data, net):
        self.data = data
        self.net = net

        self.optimizer = None
        self.batch_size = None

        self.losses = None
        self.totalloss = None
        self.train_op = None
        self.metrics = None

        self.sess = None
        self.saver = None
        self.train_state = TrainState()
        self.losshistory = LossHistory()
        self.stop_training = False
        self.callbacks = None

        self._open_tfsession()

    def close(self):
        self._close_tfsession()

    @timing
    def compile(
        self,
        optimizer,
        lr=None,
        loss="MSE",
        metrics=None,
        decay=None,
        loss_weights=None,
    ):
        """Configures the model for training.

        Args:
            optimizer: String. Name of optimizer.
            lr: A Tensor or a floating point value. The learning rate.
            loss: String (name of objective function) or objective function.
            metrics: List of metrics to be evaluated by the model during training.
            decay: String. Name of decay to the initial learning rate.
            loss_weights: A list specifying scalar coefficients (Python floats)
                to weight the loss contributions. The loss value that will be minimized by the model
                will then be the weighted sum of all individual losses,
                weighted by the loss_weights coefficients.
        """
        print("Compiling model...")

        self.optimizer = optimizer

        loss = losses_module.get(loss)
        self.losses = self.data.losses(self.net.targets, self.net.outputs, loss, self)
        if self.net.regularizer is not None:
            self.losses.append(tf.losses.get_regularization_loss())
        self.losses = tf.convert_to_tensor(self.losses)
        if loss_weights is not None:
            self.losses *= loss_weights
            self.losshistory.set_loss_weights(loss_weights)
        self.totalloss = tf.reduce_sum(self.losses)

        self.train_op = train_module.get_train_op(
            self.totalloss, self.optimizer, lr=lr, decay=decay
        )

        metrics = metrics or []
        self.metrics = [metrics_module.get(m) for m in metrics]

    @timing
    def train(
        self,
        epochs=None,
        batch_size=None,
        display_every=1000,
        uncertainty=False,
        disregard_previous_best=False,
        callbacks=None,
        model_restore_path=None,
        model_save_path=None,
        print_model=False,
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            epochs: Integer. Number of epochs to train the model.
            batch_size: Integer or ``None``. Not fully supported yet.
            display_every: Integer. Print the loss and metrics every this steps.
            uncertainty: Boolean. If ``True``, use Monte-Carlo Dropout to estimate uncertainty.
            disregard_previous_best: If ``True``, disregard the previous saved best model.
            callbacks: List of ``deepxde.callbacks.Callback`` instances.
                List of callbacks to apply during training.
            model_restore_path: String. Path where parameters were previously saved.
                See `tf.train.Saver.restore <https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/train/Saver#restore>`_.
            model_save_path: String. Prefix of filenames created for the checkpoint.
                See `tf.train.Saver.save <https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/train/Saver#save>`_.
            print_model: If ``True``, print the values of all variables.
        """
        self.batch_size = batch_size
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        if disregard_previous_best:
            self.train_state.disregard_best()

        if self.train_state.step == 0:
            print("Initializing variables...")
            self.sess.run(tf.global_variables_initializer())
        else:
            guarantee_initialized_variables(self.sess)
        if model_restore_path is not None:
            print("Restoring model from {} ...".format(model_restore_path))
            self.saver.restore(self.sess, model_restore_path)

        print("Training model...\n")
        self.stop_training = False
        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_state.set_data_test(*self.data.test())
        self._test(uncertainty)
        self.callbacks.on_train_begin()
        if train_module.is_scipy_opts(self.optimizer):
            self._train_scipy(display_every, uncertainty)
        else:
            if epochs is None:
                raise ValueError("No epochs for {}.".format(self.optimizer))
            self._train_sgd(epochs, display_every, uncertainty)
        self.callbacks.on_train_end()

        print("")
        display.training_display.summary(self.train_state)
        if print_model:
            self._print_model()
        if model_save_path is not None:
            self.save(model_save_path, verbose=1)


        return self.losshistory, self.train_state

    def evaluate(self, x, y, callbacks=None):
        """Returns the loss values & metrics values for the model in test mode.
        """
        raise NotImplementedError(
            "Model.evaluate to be implemented. Alternatively, use Model.predict."
        )

    @timing
    def predict(self, x, operator=None, callbacks=None):
        """Generates output predictions for the input samples.
        """
        print("Predicting...")
        self.callbacks = CallbackList(callbacks=callbacks)
        self.callbacks.set_model(self)
        self.callbacks.on_predict_begin()
        if operator is None:
            y = self.sess.run(
                self.net.outputs,
                feed_dict=self._get_feed_dict(False, False, 2, x, None),
            )
        else:
            y = self.sess.run(
                operator(self.net.inputs, self.net.outputs),
                feed_dict=self._get_feed_dict(False, False, 2, x, None),
            )
        self.callbacks.on_predict_end()
        return y

    def _open_tfsession(self):
        tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)
        self.saver = tf.train.Saver(max_to_keep=None)
        self.train_state.set_tfsession(self.sess)

    def _close_tfsession(self):
         self.sess.close()

    def _train_sgd(self, epochs, display_every, uncertainty):
        for i in range(epochs):
            self.callbacks.on_epoch_begin()
            self.callbacks.on_batch_begin()

            self.train_state.set_data_train(
                *self.data.train_next_batch(self.batch_size)
            )
            self.sess.run(
                self.train_op,
                feed_dict=self._get_feed_dict(
                    True, True, 0, self.train_state.X_train, self.train_state.y_train
                ),
            )

            self.train_state.epoch += 1
            self.train_state.step += 1
            if self.train_state.step % display_every == 0 or i + 1 == epochs:
                self._test(uncertainty)

            self.callbacks.on_batch_end()
            self.callbacks.on_epoch_end()

            if self.stop_training:
                break

    def _train_scipy(self, display_every, uncertainty):
        def loss_callback(loss_train):
            self.train_state.epoch += 1
            self.train_state.step += 1
            self.train_state.loss_train = loss_train
            self.train_state.loss_test = None
            self.train_state.metrics_test = None
            self.losshistory.append(
                self.train_state.step, self.train_state.loss_train, None, None
            )
            if self.train_state.step % display_every == 0:
                display.training_display(self.train_state)

        self.train_state.set_data_train(*self.data.train_next_batch(self.batch_size))
        self.train_op.minimize(
            self.sess,
            feed_dict=self._get_feed_dict(
                True, True, 0, self.train_state.X_train, self.train_state.y_train
            ),
            fetches=[self.losses],
            loss_callback=loss_callback,
        )
        self._test(uncertainty)

    def _test(self, uncertainty):
        self.train_state.loss_train, self.train_state.y_pred_train = self.sess.run(
            [self.losses, self.net.outputs],
            feed_dict=self._get_feed_dict(
                False, False, 0, self.train_state.X_train, self.train_state.y_train
            ),
        )

        if uncertainty:
            # TODO: support multi outputs
            losses, y_preds = [], []
            for _ in range(1000):
                loss_one, y_pred_test_one = self.sess.run(
                    [self.losses, self.net.outputs],
                    feed_dict=self._get_feed_dict(
                        False, True, 1, self.train_state.X_test, self.train_state.y_test
                    ),
                )
                losses.append(loss_one)
                y_preds.append(y_pred_test_one)
            self.train_state.loss_test = np.mean(losses, axis=0)
            self.train_state.y_pred_test = np.mean(y_preds, axis=0)
            self.train_state.y_std_test = np.std(y_preds, axis=0)
        else:
            self.train_state.loss_test, self.train_state.y_pred_test = self.sess.run(
                [self.losses, self.net.outputs],
                feed_dict=self._get_feed_dict(
                    False, False, 1, self.train_state.X_test, self.train_state.y_test
                ),
            )

        if isinstance(self.net.targets, (list, tuple)):
            self.train_state.metrics_test = [
                m(self.train_state.y_test[i], self.train_state.y_pred_test[i])
                for m in self.metrics
                for i in range(len(self.net.targets))
            ]
        else:
            self.train_state.metrics_test = [
                m(self.train_state.y_test, self.train_state.y_pred_test)
                for m in self.metrics
            ]

        self.train_state.update_best()
        self.losshistory.append(
            self.train_state.step,
            self.train_state.loss_train,
            self.train_state.loss_test,
            self.train_state.metrics_test,
        )
        display.training_display(self.train_state)

    def _get_feed_dict(self, training, dropout, data_id, inputs, targets):
        feed_dict = {
            self.net.training: training,
            self.net.dropout: dropout,
            self.net.data_id: data_id,
        }
        if isinstance(self.net.inputs, (list, tuple)):
            feed_dict.update(dict(zip(self.net.inputs, inputs)))
        else:
            feed_dict.update({self.net.inputs: inputs})
        if targets is None:
            return feed_dict
        if isinstance(self.net.targets, (list, tuple)):
            feed_dict.update(dict(zip(self.net.targets, targets)))
        else:
            feed_dict.update({self.net.targets: targets})
        return feed_dict

    def _print_model(self):
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}, Shape: {}".format(k, v.shape))
            print(v)

    def save(self, save_path, verbose=0):
        if verbose > 0:
            print(
                "Epoch {}: saving model to {}-{} ...\n".format(
                    self.train_state.epoch, save_path, self.train_state.epoch
                )
            )
        self.saver.save(self.sess, save_path, global_step=self.train_state.epoch)

    def restore(self, save_path, verbose=0):
        if verbose > 0:
            print("Restoring model from {} ...\n".format(save_path))
        self.saver.restore(self.sess, save_path)


class TrainState(object):
    def __init__(self):
        self.epoch, self.step = 0, 0

        self.sess = None

        # Data
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None

        # Results of current step
        self.y_pred_train = None
        self.loss_train, self.loss_test = None, None
        self.y_pred_test, self.y_std_test = None, None
        self.metrics_test = None

        # The best results correspond to the min train loss
        self.best_step = 0
        self.best_loss_train, self.best_loss_test = np.inf, np.inf
        self.best_y, self.best_ystd = None, None
        self.best_metrics = None

    def set_tfsession(self, sess):
        self.sess = sess

    def set_data_train(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train

    def set_data_test(self, X_test, y_test):
        self.X_test, self.y_test = X_test, y_test

    def update_best(self):
        if self.best_loss_train > np.sum(self.loss_train):
            self.best_step = self.step
            self.best_loss_train = np.sum(self.loss_train)
            self.best_loss_test = np.sum(self.loss_test)
            self.best_y, self.best_ystd = self.y_pred_test, self.y_std_test
            self.best_metrics = self.metrics_test

    def disregard_best(self):
        self.best_loss_train = np.inf

    def packed_data(self):
        def merge_values(values):
            return np.hstack(values) if isinstance(values, (list, tuple)) else values

        X_train = merge_values(self.X_train)
        y_train = merge_values(self.y_train)
        X_test = merge_values(self.X_test)
        y_test = merge_values(self.y_test)
        best_y = merge_values(self.best_y)
        best_ystd = merge_values(self.best_ystd)
        return X_train, y_train, X_test, y_test, best_y, best_ystd


class LossHistory(object):
    def __init__(self):
        self.steps = []
        self.loss_train = []
        self.loss_test = []
        self.metrics_test = []
        self.loss_weights = 1

    def set_loss_weights(self, loss_weights):
        self.loss_weights = loss_weights

    def append(self, step, loss_train, loss_test, metrics_test):
        self.steps.append(step)
        self.loss_train.append(loss_train)
        if loss_test is None:
            loss_test = self.loss_test[-1]
        if metrics_test is None:
            metrics_test = self.metrics_test[-1]
        self.loss_test.append(loss_test)
        self.metrics_test.append(metrics_test)
