from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import deepxde as dde


def gen_traindata():
    data = np.load("dataset/reaction.npz")
    t, x, ca, cb = data["t"], data["x"], data["Ca"], data["Cb"]
    X, T = np.meshgrid(x, t)
    X = np.reshape(X, (-1, 1))
    T = np.reshape(T, (-1, 1))
    Ca = np.reshape(ca, (-1, 1))
    Cb = np.reshape(cb, (-1, 1))
    return np.hstack((X, T)), Ca, Cb


def main():
    kf = tf.Variable(0.05)
    D = tf.Variable(1.0)

    def pde(x, y):
        ca, cb = y[:, 0:1], y[:, 1:2]
        ca_x = tf.gradients(ca, x)[0]
        dca_x, dca_t = ca_x[:, 0:1], ca_x[:, 1:2]
        dca_xx = tf.gradients(dca_x, x)[0][:, 0:1]
        cb_x = tf.gradients(cb, x)[0]
        dcb_x, dcb_t = cb_x[:, 0:1], cb_x[:, 1:2]
        dcb_xx = tf.gradients(dcb_x, x)[0][:, 0:1]
        eq_a = dca_t - 1e-3 * D * dca_xx + kf * ca * cb ** 2
        eq_b = dcb_t - 1e-3 * D * dcb_xx + 2 * kf * ca * cb ** 2
        return [eq_a, eq_b]

    def fun_bc(x):
        return 1 - x[:, 0:1]

    def fun_init(x):
        return np.exp(-20 * x[:, 0:1])

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 10)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc_a = dde.DirichletBC(
        geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=0
    )
    bc_b = dde.DirichletBC(
        geomtime, fun_bc, lambda _, on_boundary: on_boundary, component=1
    )
    ic1 = dde.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=0)
    ic2 = dde.IC(geomtime, fun_init, lambda _, on_initial: on_initial, component=1)

    observe_x, Ca, Cb = gen_traindata()
    ptset = dde.bc.PointSet(observe_x)
    observe_y1 = dde.DirichletBC(
        geomtime, ptset.values_to_func(Ca), lambda x, _: ptset.inside(x), component=0
    )
    observe_y2 = dde.DirichletBC(
        geomtime, ptset.values_to_func(Cb), lambda x, _: ptset.inside(x), component=1
    )

    data = dde.data.TimePDE(
        geomtime,
        2,
        pde,
        [bc_a, bc_b, ic1, ic2, observe_y1, observe_y2],
        num_domain=2000,
        num_boundary=100,
        num_initial=100,
        anchors=observe_x,
        num_test=50000,
    )
    net = dde.maps.FNN([2] + [20] * 3 + [2], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    variable = dde.callbacks.VariableValue(
        [kf, D], period=1000, filename="variables.dat"
    )
    losshistory, train_state = model.train(epochs=80000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)


if __name__ == "__main__":
    main()
