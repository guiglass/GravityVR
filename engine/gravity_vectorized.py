#! /usr/bin/python

#--------------------------------#
# Physics example running that manipulates the arrays for scene objects.
# Demonstrates Newton's law of gravitation in a planetary orbit simulator.
#--------------------------------#

import time
import numpy as np

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
        self.__reset_universe__()

    def update(self):
        t = 0.01 * self.time_scale #The time step scale value
        self.simTotalTime += t #second

        self.verts_coord = self._update_vectorized(t)
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

    def _update_vectorized(self, t):
        ax0, ax1, ax2 = 0, 1, 2  # allows the ability to select which axes (plane) we want to use (basically X=0,Y=1 or Y=1,Z=2 and so on..)
        mat_loc = self.verts_coord

        n = mat_loc.shape[0]
        mask = ~np.eye(n, dtype=bool)
        loc = np.broadcast_to(mat_loc, (n, n, mat_loc.shape[-1]))[mask].reshape(n, n - 1, -1)

        repeater = np.repeat(self.verts_coord, n-1, axis=0).reshape(loc.shape[0], n-1, 3)
        mat_slope = loc - repeater # rise and run -or- delta positions #todo fix sign/order??

        mat_d_z = mat_slope[:,::,ax2:ax2+1] # delta height
        mat_d_y = mat_slope[:,::,ax1:ax1+1] # rise / delta Y
        mat_d_x = mat_slope[:,::,ax0:ax0+1] # run / delta X

        mat_base = np.sqrt(mat_d_x ** 2 + mat_d_y ** 2) # the "base" lenght of the triangle which describes the change in z
        mat_hyp = np.sqrt(mat_base ** 2 + mat_d_z ** 2) # the actual "radius" length from the base vertex "verts_coord[i]" to all other vertexes

        mat_az_tan = np.arctan2(mat_d_y, mat_d_x)  # radians on the x,y plane formed from our base vertex to the other vertex
        mat_elv_tan = np.arctan2(mat_d_z, mat_base) # elevation radians -or- the angle formed from the base triangle to the tip of the gforce vector

        space = mat_hyp ** 2
        force = self.G * self.verts_mass
        mat_g = np.transpose(force / np.transpose(space)) #Newton's gravity equation

        mat_base2 = mat_g * np.cos(mat_elv_tan) #a vector residing only on the x,y plane denoting the base length of the gforce vector triangle

        mat_g_x = mat_base2 * np.cos(mat_az_tan)
        mat_g_y = mat_base2 * np.sin(mat_az_tan)
        mat_g_z = mat_g * np.sin(mat_elv_tan)  # the height (from a point rotating around the x,y plane this vector goes straight up z and is connected to the tip of gforce vector)

        sort = np.nonzero(mask)[1].argsort()
        mat_g_x = mat_g_x.ravel()[sort].reshape(mat_g_x.shape)
        mat_g_y = mat_g_y.ravel()[sort].reshape(mat_g_y.shape)
        mat_g_z = mat_g_z.ravel()[sort].reshape(mat_g_z.shape)

        mat_axis_gforce = np.sum(np.dstack((mat_g_x, mat_g_y, mat_g_z)), axis=1)

        #now apply the gforce vectors to the actual coordinate's positions and velocities
        self.verts_vel += mat_axis_gforce * t
        self.verts_coord -= self.verts_vel * t

        return self.verts_coord


    def _particle_vectorized(self, t):

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
        uncollided = mat_hyp - self.verts_radius.reshape(self.verts_radius.shape[0],1) #subtract the particles's coord from the distance to the body
        uncollided = np.prod(uncollided, axis=1) #get product for each particle's rows (if there are zeros anywhere then result will be zero)
        indeces_uncollided = np.prod(uncollided, axis=1).clip(0).nonzero() #get only the indexes of the verts that are not zero as the indices to keep

        #Uncomment this section to use fancy indexing to remove the collided elements and resize the arrays.
        #Note: this may cause a jump in speed (as cpu load drops) noticable if many particles are removed suddenly.
        #self.parts_coord = self.parts_coord[indeces_uncollided]
        #self.parts_color = self.parts_color[indeces_uncollided]
        #self.parts_radius = self.parts_radius[indeces_uncollided]
        #self.parts_vel = self.parts_vel[indeces_uncollided]

        #Otherwise use this section to just set everything to zery so particles are invisible (but will still be processed and initial array size never changes)
        indeces_collided = np.delete(np.arange(self.parts_coord.shape[0]), indeces_uncollided)
        self.parts_coord[indeces_collided] *= 0
        self.parts_color[indeces_collided] *= 0
        self.parts_radius[indeces_collided] *= 0
        self.parts_vel[indeces_collided] *= 0

        return self.parts_coord


    def _update_nonvectorized(self, t):
        #This is the non-vectorized version of _update_vectorized and is here to simply demonstrate the concept.
        ax0, ax1, ax2 = 0, 1, 2  # allows the ability to select which axes (plane) we want to use (basically X=0,Y=1 or Y=1,Z=2 and so on..)

        for i, pt in enumerate(self.verts_coord):
            loc = self.verts_coord[i] #* self.TimeScale
            rng = np.arange(self.verts_coord.shape[0])
            rng_without_num = np.append(rng[:i], rng[i + 1:], axis=0)  # get all verts except this (loc[i])

            axis_g = np.array([0, 0, 0], dtype=np.float64)

            for i2 in rng_without_num:
                _loc = self.verts_coord[i2]

                d_z = _loc[ax2] - loc[ax2]  # delta height
                d_y = _loc[ax1] - loc[ax1]  # rise / delta Y
                d_x = _loc[ax0] - loc[ax0]  # run / delta X

                base = np.sqrt(d_x ** 2 + d_y ** 2) # the "base" lenght of the triangle which describes the change in z
                hyp = np.sqrt(base ** 2 + d_z ** 2) # the actual "radius" length from the base vertex "verts_coord[i]" to all other vertexes as iterated from "rng_without_num[i2]"

                az_tan = np.arctan2(d_y,d_x)  # radians on the x,y plane formed from our base vertex to the other vertex as iterated from "rng_without_num[i2]"
                elv_tan = np.arctan2(d_z,base)  # radians on the of the height formed from the base triangle and a height

                mass = self.verts_mass[i2]  # the mass of the object in "rng_without_num"

                space = hyp ** 2
                force = self.G * mass

                g = force / space
                b2 = g * np.cos(elv_tan)  # a vector residing only on the x,y plane denoting the base length of the gforce vector triangle

                g_x = b2 * np.cos(az_tan)
                g_y = b2 * np.sin(az_tan)
                g_z = g * np.sin(elv_tan)  # the height (from a point rotating around the x,y plane this vector goes straight up z and is connected to the tip of gforce vector)

                axis_g += np.array([g_x, g_y, g_z])

            self.verts_coord[i] += axis_g * t
            self.verts_coord[i] += self.verts_coord[i] * t

        return self.verts_coord


    def _particle_nonvectorized(self, t):
        #This is the non-vectorized version of _particle_vectorized and is here to simply demonstrate the concept.
        ax0, ax1, ax2 = 0, 1, 2  # allows the ability to select which axes (plane) we want to use (basically X=0,Y=1 or Y=1,Z=2 and so on..)
        remove_verts = np.array([])

        for i, loc in enumerate(self.parts_coord):
            rng = np.arange(self.verts_coord.shape[0])
            axis_g = np.array([0, 0, 0], dtype=np.float64)
            for i2 in rng:
                _loc = self.verts_coord[i2]

                d_z = _loc[ax2] - loc[ax2]  # delta height
                d_y = _loc[ax1] - loc[ax1]  # rise / delta Y
                d_x = _loc[ax0] - loc[ax0]  # run / delta X

                base = np.sqrt(d_x ** 2 + d_y ** 2) # the "base" lenght of the triangle which describes the change in z
                hyp = np.sqrt(base ** 2 + d_z ** 2) # the actual "radius" length from the base vertex "verts_coord[i]" to all other vertexes as iterated from "rng_without_num[i2]"

                #collision detection
                if hyp - self.verts_radius[i2] <= self.parts_radius[i]:
                    remove_verts = np.append(remove_verts, i) #comment out this line for bouncy behavior instead of jsut killing the particle
                    continue

                az_tan = np.arctan2(d_y,d_x)  # radians on the x,y plane formed from our base vertex to the other vertex as iterated from "rng_without_num[i2]"
                elv_tan = np.arctan2(d_z,base)  # radians on the of the height formed from the base triangle and a height

                mass = self.verts_mass[i2]  # the mass of the object in "rng_without_num"

                space = hyp ** 2
                force = self.G * mass

                g = force / space
                b2 = g * np.cos(elv_tan)  # a vector residing only on the x,y plane denoting the base length of the gforce vector triangle

                g_x = b2 * np.cos(az_tan)
                g_y = b2 * np.sin(az_tan)
                g_z = g * np.sin(elv_tan)  # the height (from a point rotating around the x,y plane this vector goes straight up z and is connected to the tip of gforce vector)

                axis_g += np.array([g_x, g_y, g_z])

            #now apply the gforce vectors to the actual coordinate's positions and velocities
            self.parts_vel[i] += axis_g * t
            self.parts_coord[i] += self.parts_vel[i] * t

        self.parts_coord = np.delete(self.parts_coord, remove_verts, axis=0)
        self.parts_color = np.delete(self.parts_color, remove_verts, axis=0)
        self.parts_vel = np.delete(self.parts_vel, remove_verts, axis=0)
        self.parts_radius = np.delete(self.parts_radius, remove_verts, axis=0)

        return self.parts_coord

