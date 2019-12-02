"""Operations which handle numpy and tensorflow automatically."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from . import config


def istensor(value):
    return isinstance(value, (tf.Tensor, tf.Variable, tf.SparseTensor))


def istensorlist(values):
    return any(map(istensor, values))


def shape(value):
    return value.get_shape() if istensor(value) else value.shape


def convert_to_array(value):
    """Convert a list to numpy array or tensorflow tensor."""
    if istensorlist(value):
        return tf.convert_to_tensor(value, dtype=config.real(tf))
    value = np.array(value)
    if value.dtype != config.real(np):
        return value.astype(config.real(np))
    return value


def hstack(tup):
    if tup[0] == []:
        tup = list(tup)
        if istensorlist(tup[1:]):
            tup[0] = tf.convert_to_tensor([], dtype=config.real(tf))
        else:
            tup[0] = np.array([], dtype=config.real(np))
    return tf.concat(tup, 0) if istensor(tup[0]) else np.hstack(tup)
