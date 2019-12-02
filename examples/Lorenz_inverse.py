from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import deepxde as dde


def gen_traindata():
    data = np.load("dataset/Lorenz.npz")
    return data["t"], data["y"]


def main():
    C1 = tf.Variable(1.0)
    C2 = tf.Variable(1.0)
    C3 = tf.Variable(1.0)

    def Lorenz_system(x, y):
        """Lorenz system.
        dy1/dx = 10 * (y2 - y1)
        dy2/dx = y1 * (28 - y3) - y2
        dy3/dx = y1 * y2 - 8/3 * y3
        """
        y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
        dy1_x = tf.gradients(y1, x)[0]
        dy2_x = tf.gradients(y2, x)[0]
        dy3_x = tf.gradients(y3, x)[0]
        return [
            dy1_x - C1 * (y2 - y1),
            dy2_x - y1 * (C2 - y3) + y2,
            dy3_x - y1 * y2 + C3 * y3,
        ]

    def boundary(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)

    geom = dde.geometry.Interval(0, 3)

    # Initial conditions
    ic1 = dde.DirichletBC(geom, lambda X: -8 * np.ones(X.shape), boundary, component=0)
    ic2 = dde.DirichletBC(geom, lambda X: 7 * np.ones(X.shape), boundary, component=1)
    ic3 = dde.DirichletBC(geom, lambda X: 27 * np.ones(X.shape), boundary, component=2)

    # Get the train data
    observe_t, ob_y = gen_traindata()
    ptset = dde.bc.PointSet(observe_t)
    inside = lambda x, _: ptset.inside(x)
    observe_y0 = dde.DirichletBC(
        geom, ptset.values_to_func(ob_y[:, 0:1]), inside, component=0
    )
    observe_y1 = dde.DirichletBC(
        geom, ptset.values_to_func(ob_y[:, 1:2]), inside, component=1
    )
    observe_y2 = dde.DirichletBC(
        geom, ptset.values_to_func(ob_y[:, 2:3]), inside, component=2
    )

    data = dde.data.PDE(
        geom,
        3,
        Lorenz_system,
        [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
        num_domain=400,
        num_boundary=2,
        anchors=observe_t,
    )

    net = dde.maps.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    variable = dde.callbacks.VariableValue(
        [C1, C2, C3], period=600, filename="variables.dat"
    )
    losshistory, train_state = model.train(epochs=60000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
