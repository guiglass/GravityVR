import numpy as np
from .planet_params import *
from .planetary_rings import get_rings
from .orbital_velocity import get_orbital_velocity
from .rotation_matrix import rotation_matrix

def create_bodies(parent, pos, vel, mass, radius, color):
    if parent.builder.verts_coord is None:
        parent.builder.verts_coord = np.array([pos], dtype=np.float)
        parent.builder.verts_vel = np.array([vel], dtype=np.float)
        parent.builder.verts_mass = np.array([mass], dtype=np.float)
        parent.builder.verts_radius = np.array([radius])
        parent.builder.verts_color = np.array([color])
    else:
        s = parent.builder.verts_coord.shape[0]
        parent.builder.verts_coord = np.concatenate((parent.builder.verts_coord.reshape(s,3), np.array(pos).reshape(1,3)), axis=0)
        parent.builder.verts_vel = np.concatenate((parent.builder.verts_vel.reshape(s,3), np.array(vel).reshape(1,3)), axis=0)
        parent.builder.verts_mass = np.concatenate((parent.builder.verts_mass, np.array([mass])), axis=0)
        parent.builder.verts_radius = np.concatenate((parent.builder.verts_radius, np.array([radius])), axis=0)
        parent.builder.verts_color = np.concatenate((parent.builder.verts_color.reshape(s,4), np.array(color).reshape(1,4)), axis=0)
    return parent.builder.verts_coord.shape[0]-1 #the index (id) of this body (used to store this model instance's array index)

def create_particles(parent, coord, velocity, radius, color):
    if parent.builder.parts_coord is None:
        parent.builder.parts_coord = coord
        parent.builder.parts_vel = velocity
        parent.builder.parts_radius = radius
        parent.builder.parts_color = color
    else:
        parent.builder.parts_coord = np.concatenate((parent.builder.parts_coord, coord), axis=0)
        parent.builder.parts_vel = np.concatenate((parent.builder.parts_vel, velocity), axis=0)
        parent.builder.parts_radius = np.concatenate((parent.builder.parts_radius, radius), axis=0)
        parent.builder.parts_color = np.concatenate((parent.builder.parts_color, color), axis=0)

class Sun():

    def __init__(self, builder):
        self.builder = builder

    def create(self, pos=(0, 0, 0), vel=(0, 0, 0)):
        sun = create_bodies(self, pos, vel, MassSun, RadiusSun, ColorSun)

class Mercury():

    def __init__(self, builder):
        self.builder = builder

    def create(self, pos=(OrbitMercurySun, 0, 0), vel=(0, 0, OrbitMercurySunVelocity)):
        mercury = create_bodies(self, pos, vel, MassMercury, RadiusMercury, ColorMercury)

class Venus():

    def __init__(self, builder):
        self.builder = builder

    def create(self, pos=(OrbitVenusSun, 0, 0), vel=(0, 0, OrbitVenusSunVelocity)):
        venus = create_bodies(self, pos, vel, MassVenus, RadiusVenus, ColorVenus)

class Earth():

    def __init__(self, builder):
        self.builder = builder

    def create(self, pos=(OrbitEarthSun, 0, 0), vel=(0, 0, OrbitEarthSunVelocity)):
        earth = create_bodies(self, pos, vel, MassEarth, RadiusEarth, ColorEarth)
        moon = create_bodies(self, pos, vel, MassMoon, RadiusMoon, ColorMoon)
        self.builder.verts_coord[moon][0] += OrbitEarthMoon
        self.builder.verts_vel[moon] = self.builder.verts_vel[earth].astype(np.float)

        #calculate the tilted axis of lunar orbit around earth
        axis = [1, 0, 0]
        theta = np.radians(5.14)
        v = get_orbital_velocity(self.builder.verts_coord[earth], self.builder.verts_mass[earth], self.builder.verts_coord[moon])
        v = np.dot(rotation_matrix(axis, theta), v)
        self.builder.verts_vel[moon] += v
        c = np.dot(rotation_matrix(axis, theta), self.builder.verts_coord[moon])
        self.builder.verts_coord[moon] = c

class Mars():

    def __init__(self, builder):
        self.builder = builder

    def create(self, pos=(OrbitMarsSun, 0, 0), vel=(0, 0, OrbitMarsSunVelocity)):
        mars = create_bodies(self, pos, vel, MassMars, RadiusMars, ColorMars)

class Jupiter():

    def __init__(self, builder):
        self.builder = builder

    def create(self, pos=(OrbitJupiterSun, 0, 0), vel=(0, 0, OrbitJupiterSunVelocity)):
        jupiter = create_bodies(self, pos, vel, MassJupiter, RadiusJupiter, ColorJupiter)

class Saturn():

    #Default params for Saturn's rings
    axis=[1, 0, 0]
    theta=np.radians(45)
    ring_groups = 4
    ring_bands = 10
    min_rad=4 * 10 ** 7
    max_rad=8 * 10 ** 7
    particle_size=5 * 10 ** 5
    n_particles = 2000 #default num of particles (must be greater than or equal to ring_groups*ring_bands)
    particle_color = np.array((
        np.ones([ring_bands, 4]) * ColorSaturnRing4,
        np.ones([ring_bands, 4]) * ColorSaturnRing3,
        np.ones([ring_bands, 4]) * ColorSaturnRing2,
        np.ones([ring_bands, 4]) * ColorSaturnRing1
    ))

    def __init__(self, builder):
        self.builder = builder

    def create(self, pos=(OrbitSaturnSun, 0, 0), vel=(0, 0, OrbitSaturnSunVelocity)):
        saturn = create_bodies(self, pos, vel, MassSaturn, RadiusSaturn, ColorSaturn)

        #Build the rings
        rings=int(self.ring_groups * self.ring_bands)
        parts_per_ring = int(self.n_particles / self.ring_groups / self.ring_bands)
        for x in range(rings):
            color = self.particle_color.reshape(rings,4)[x]#self.particle_color[x % self.particle_color.shape[0]] #cycle through colors (if list of colors was passed)
            ring_rad = (self.max_rad - self.min_rad) / (rings/(x+1))

            create_particles(self, *get_rings(self.builder.verts_radius[-1] + ring_rad, self.builder.verts_coord[-1], vel, self.builder.verts_mass[-1], n_particles=parts_per_ring, particle_size=self.particle_size, particle_color=color, axis=self.axis, theta=self.theta))

