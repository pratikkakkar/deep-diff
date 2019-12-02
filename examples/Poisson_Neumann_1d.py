from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import deepxde as dde


def main():
    def pde(x, y):
        dy_x = tf.gradients(y, x)[0]
        dy_xx = tf.gradients(dy_x, x)[0]
        return dy_xx - 2

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], -1)

    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], 1)

    def func(x):
        return (x + 1) ** 2

    geom = dde.geometry.Interval(-1, 1)
    bc_l = dde.DirichletBC(geom, func, boundary_l)
    bc_r = dde.NeumannBC(geom, lambda X: 2 * (X + 1), boundary_r)
    data = dde.data.PDE(geom, 1, pde, [bc_l, bc_r], 16, 2, func=func, num_test=100)

    layer_size = [1] + [50] * 3 + [1]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=10000)

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
