#! /usr/bin/python

#--------------------------------#
# Physics example running in an openGL PyQtGraph Scatterplot.
# Demonstrates Newton's law of gravitation in a planetary orbit simulator.
#--------------------------------#

import sys
import time
import numpy as np
from datetime import timedelta

import tensorflow as tf

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl


class newtonianLawOfGravitation():
    builder = None #the builder object for the scene actor

    verts_coord = None #vertex coordinates
    verts_color = None
    verts_radius = None

    parts_coord = None #particle coordinates
    parts_color = None
    parts_radius = None

    sess = None

    size_scale = 1 * 10 ** 8 #defaults to 1 million kilometers per unit
    time_scale = 1 #defaults to 1 but can be adjusted with slider control

    G = 6.674 * 10 ** -11  # Newton meters^2/kg^2

    def __init__(self, builder):
        self._builder = builder
        self.sess = tf.Session()
        self.__reset_universe__()

        self.sess.run(tf.global_variables_initializer())

    def update(self):
        t = 0.01 * self.time_scale #The time step scale value
        self.simTotalTime += t # second

        self.verts_coord = self._update_tensorflow(t)
        if self.builder.parts_coord is not None:
            particles = self._particle_vectorized(t)
            vretices = np.append(self.verts_coord, particles, axis=0) / self.size_scale
            colors = self.colors = np.append(self.verts_color, self.parts_color, axis=0)
            return vretices, colors
        else:
            vretices = self.verts_coord / self.size_scale
            colors = self.verts_color
            return vretices, colors


    def __reset_timers__(self):
        self.simStartTime = time.time()
        self.simLastTime = self.simStartTime
        self.simTotalTime = 0

    def __reset_universe__(self):
        self.__load_builder__()
        self.__reset_timers__()
        self.__init_tensorflow_graph()

    def __load_builder__(self):
        self.builder = self._builder(self.size_scale)

        self.verts_coord  = self.builder.verts_coord
        self.verts_radius = self.builder.verts_radius
        self.verts_color  = self.builder.verts_color
        self.verts_vel    = self.builder.verts_vel
        self.verts_mass   = self.builder.verts_mass

        if self.builder.parts_coord is not None:
            epsilon = 0.000001 #helps avoid divide by zero errors on first cycle
            self.parts_coord  = (self.builder.parts_coord + epsilon)
            self.parts_radius = self.builder.parts_radius
            self.parts_color  = self.builder.parts_color
            self.parts_vel    = self.builder.parts_vel


    def __init_tensorflow_graph(self):
        if self.sess is not None:
            self.sess.close()
        self.sess = tf.Session()

        def atan2_tensor(y, x):
            angle = tf.where(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
            angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.greater_equal(y, 0.0)), tf.atan(y / x) + np.pi, angle)
            angle = tf.where(tf.logical_and(tf.less(x, 0.0), tf.less(y, 0.0)), tf.atan(y / x) - np.pi, angle)
            angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.greater(y, 0.0)), 0.5 * np.pi * tf.ones_like(x), angle)
            angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.less(y, 0.0)), -0.5 * np.pi * tf.ones_like(x), angle)
            angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)), tf.zeros_like(x), angle)
            return angle

        ax0, ax1, ax2 = 0, 1, 2

        self.ts = tf.placeholder(tf.float64, shape=())

        self.tensor_coord = tf.Variable(self.verts_coord, dtype=tf.float64)
        self.vel = tf.Variable(self.verts_vel, dtype=tf.float64)
        mass = tf.Variable(self.verts_mass, dtype=tf.float64)

        n = self.tensor_coord .get_shape()[0].value
        eye = ~np.eye(n, dtype=bool)

        tile = tf.tile(self.tensor_coord, [n, 1])
        tile = tf.reshape(tile, [n, n, -1])

        mask = tf.constant(eye)
        loc = tf.boolean_mask(tile, mask)
        loc = tf.reshape(loc, [n, n - 1, 3])

        rep = tf.tile(self.tensor_coord, [1, n - 1])
        rep = tf.reshape(rep, [n, n - 1, 3])

        slope = tf.subtract(loc, rep)  # rise and run -or- delta positions
        d_z = slope[:, ::, ax2:ax2 + 1]  # delta height
        d_y = slope[:, ::, ax1:ax1 + 1]  # rise / delta Y
        d_x = slope[:, ::, ax0:ax0 + 1]  # run / delta X

        base = tf.sqrt(tf.add(tf.pow(d_x, 2), tf.pow(d_y, 2)))  # the "base" lenght of the triangle which describes the change in z
        hyp = tf.sqrt(tf.add(tf.pow(base, 2), tf.pow(d_z, 2)))
        #hyp = tf.multiply(hyp, self.size_scale)  # the actual "radius" length from the base vertex "verts_coord[i]" to all other vertexes

        az_tan = atan2_tensor(d_y, d_x)  # radians on the x,y plane formed from our base vertex to the other vertex
        elv_tan = atan2_tensor(d_z,base)  # elevation radians -or- the angle formed from the base triangle to the tip of the gforce vector

        space = tf.pow(hyp, 2)
        force = tf.multiply(mass, self.G)

        g = tf.transpose(tf.divide(force, tf.transpose(space)))

        base2 = tf.multiply(g, tf.cos(elv_tan))  # a vector residing only on the x,y plane denoting the base length of the gforce vector triangle

        g_x = tf.multiply(base2, tf.cos(az_tan))
        g_y = tf.multiply(base2, tf.sin(az_tan))
        g_z = tf.multiply(g, tf.sin(elv_tan))  # the height (from a point rotating around the x,y plane this vector goes straight up z and is connected to the tip of gforce vector)

        sort = tf.constant(np.nonzero(eye)[1].argsort())
        g_x = tf.reshape(tf.gather(tf.reshape(g_x, [-1]), sort), g_x.get_shape())
        g_y = tf.reshape(tf.gather(tf.reshape(g_y, [-1]), sort), g_y.get_shape())
        g_z = tf.reshape(tf.gather(tf.reshape(g_z, [-1]), sort), g_z.get_shape())

        g_x = tf.reduce_sum(g_x, 1)
        g_y = tf.reduce_sum(g_y, 1)
        g_z = tf.reduce_sum(g_z, 1)

        axis_gforce = tf.stack([g_x, g_y, g_z], 2)[:, 0]

        # now apply the gforce vectors to the actual coordinate's positions and velocities
        self.vel = tf.assign(self.vel, tf.add(self.vel, tf.multiply(axis_gforce, self.ts)))
        self.tensor_coord = tf.assign(self.tensor_coord, tf.subtract(self.tensor_coord, tf.multiply(self.vel, self.ts)))

        self.sess.run(tf.global_variables_initializer())

    def _update_tensorflow(self, t):
        return self.sess.run(self.tensor_coord, feed_dict={self.ts: t})


    def _particle_vectorized(self, t):
        #todo someday this may also be done using Tensorflow
        ax0, ax1, ax2 = 0, 1, 2  # allows the ability to select which axes (plane) we want to use (basically X=0,Y=1 or Y=1,Z=2 and so on..)

        loc = np.tile(self.verts_coord, (self.parts_coord.shape[0], 1, 1))

        i2 = self.verts_coord.shape[0]
        repeater = np.repeat(self.parts_coord, i2, axis=0).reshape(loc.shape[0], i2, 3) #the mat for each particle to calc against each vert

        mat_slope = loc - repeater # rise and run -or- delta positions

        mat_d_z = mat_slope[:,::,ax2:ax2+1] # delta height
        mat_d_y = mat_slope[:,::,ax1:ax1+1] # rise / delta Y
        mat_d_x = mat_slope[:,::,ax0:ax0+1] # run / delta X

        mat_base = np.sqrt(mat_d_x ** 2 + mat_d_y ** 2) # the "base" lenght of the triangle which describes the change in z
        mat_hyp = np.sqrt(mat_base ** 2 + mat_d_z ** 2) # the actual "radius" length from the base vertex "verts_coord[i]" to all other vertexes

        mat_az_tan = np.arctan2(mat_d_y, mat_d_x)  # radians on the x,y plane formed from our base vertex to the other vertex
        mat_elv_tan = np.arctan2(mat_d_z, mat_base) # elevation radians -or- the angle formed from the base triangle to the tip of the gforce vector

        space = mat_hyp ** 2
        force = self.G * self.verts_mass
        force = np.tile(force, space.shape[0]).reshape(space.shape) #prepare for opperaton on all mat_space rows

        mat_g = force / space #Newton's gravity equation
        mat_base2 = mat_g * np.cos(mat_elv_tan) #a vector residing only on the x,y plane denoting the base length of the gforce vector triangle

        mat_g_x = mat_base2 * np.cos(mat_az_tan)
        mat_g_y = mat_base2 * np.sin(mat_az_tan)
        mat_g_z = mat_g * np.sin(mat_elv_tan)  # the height (from a point rotating around the x,y plane this vector goes straight up z and is connected to the tip of gforce vector)

        mat_axis_gforce = np.sum(np.dstack((mat_g_x, mat_g_y, mat_g_z)), axis=1)

        self.parts_vel += mat_axis_gforce * t
        self.parts_coord += self.parts_vel * t

        #now do collision detection and keep only particles that have not collided with any bodies
        collisions = mat_hyp - self.verts_radius.reshape(self.verts_radius.shape[0],1) #subtract the particles's coord from the distance to the body
        collisions = np.prod(collisions, axis=1) #get product for each particle's rows (if there are zeros anywhere then result will be zero)
        collisions = np.prod(collisions, axis=1).clip(0).nonzero() #get only the indexes of the verts that are not zero as the indices to keep

        self.parts_coord = self.parts_coord[collisions]
        self.parts_color = self.parts_color[collisions]
        self.parts_radius = self.parts_radius[collisions]
        self.parts_vel = self.parts_vel[collisions]

        return self.parts_coord


