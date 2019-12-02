# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:21:27 2019

@author: z0043abh
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import deepxde as dde
import matplotlib.pyplot as plt


def pde(x, u):
        du = tf.gradients(u, x)[0]
        du_x, du_y, du_z, du_t = du[:, 0:1], du[:, 1:2], du[:, 2:3], du[:, 3:4]
        du_xx = tf.gradients(du_x, x)[0][:, 0:1]
        du_yy = tf.gradients(du_y, x)[0][:, 1:2]
        du_zz = tf.gradients(du_z, x)[0][:, 2:3]
        return du_t - 0.2*(du_xx+du_yy+du_zz)

geom = dde.geometry.geometry_3d.Cuboid([0,0,0],[1,1,1])
timedomain = dde.geometry.TimeDomain(0,1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.DirichletBC(
    geomtime, lambda x: np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary
)

ic = dde.IC(
    geomtime, lambda x: 100*x[:, 0:1]*(1-x[:, 0:1])*x[:, 1:2]*(1-x[:, 1:2])*x[:, 2:3]*(1-x[:, 2:3]), lambda _, on_initial: on_initial
)

data = dde.data.TimePDE(geomtime, 1, pde, [bc,ic], num_domain=60, num_boundary=20, num_initial=20)

net = dde.maps.FNN([4] + [10] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)

model.compile("adam", lr=1e-3)
model.train(epochs=1000)
model.compile("L-BFGS-B")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
z = np.linspace(0, 1, 10)
t = np.linspace(0, 1, 10)

xx, yy, zz, tt = np.meshgrid(x, y, z, t)
data = np.vstack((np.ravel(xx), np.ravel(yy), np.ravel(zz), np.ravel(tt))).T


u = model.predict(data)


U=np.reshape(u,(10,10,10,10))

for k in range(0,10):
    im1 = plt.imshow(U[5,:,:,k], cmap=plt.cm.RdBu, extent=(0, 1,0, 1),interpolation='bilinear')
    plt.clim(0,0.5)
    plt.colorbar(im1)
    plt.show()
