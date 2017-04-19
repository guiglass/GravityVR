import numpy as np
from .extras.planet_models import *
from .extras.planet_params import *

#Each Scene_ class is to be loaded as a prebuild scene to be displayed in the 3d window.
#They construct a the vertices, colors, sizes and velocities for various simulations.

def get_scene_list():
    return (
        ("1. Simple Solar System", Scene_SolarSystem),
        ("2. Saturn Vs. Jupiter", Scene_SaturnVsJupiter),
        ("3. Random Massive Spheres", Scene_RandomSpheres),
    )

class Scene_SolarSystem():
    # Planet Starting Positions
    _coordinates = np.array([])
    _particles = np.array([]) #Optional

    verts_coord = None
    verts_radius = None
    verts_color = None
    verts_vel = None
    verts_mass = None

    parts_coord = None
    parts_radius = None
    parts_color = None
    parts_vel = None

    def __init__(self, size_scale):
        self.size_scale = size_scale

        Sun(self).create()
        Mercury(self).create()
        Venus(self).create()
        Earth(self).create()
        Jupiter(self).create()
        Saturn(self).create()

    def get_array_size(self):
        if self.parts_coord is not None:
            return self.verts_coord.shape[0] + self.parts_coord.shape[0]
        return self.verts_coord.shape[0]

class Scene_SaturnVsJupiter():
    # Planet Starting Positions
    _coordinates = np.array([])
    _particles = np.array([]) #Optional

    verts_coord = None
    verts_radius = None
    verts_color = None
    verts_vel = None
    verts_mass = None

    parts_coord = None
    parts_radius = None
    parts_color = None
    parts_vel = None

    def __init__(self, size_scale):
        self.size_scale = size_scale

        saturn = Saturn(self)
        saturn.n_particles = 10000
        saturn.create(
            pos=(0,0,0),
            vel=(0,0,0),
        )

        Jupiter(self).create(
            pos=(1*10**9,0,1.0*10**9),
            vel=(7500,0,0),
        )

    def get_array_size(self):
        if self.parts_coord is not None:
            return self.verts_coord.shape[0] + self.parts_coord.shape[0]
        return self.verts_coord.shape[0]

class Scene_RandomSpheres():

    n_particles = 200
    n_bodies = 100

    verts_coord = None
    verts_radius = None
    verts_color = None
    verts_vel = None
    verts_mass = None

    parts_coord = None
    parts_radius = None
    parts_color = None
    parts_vel = None

    def __init__(self, size_scale):
        self.size_scale = size_scale
        self.__initialize_arrays__()

    def __initialize_arrays__(self):
        self.verts_coord = self.generate_rand_coordinates(self.n_bodies)#coordinates of vertices
        self.verts_vel = np.zeros_like(self.verts_coord) #velocity of each vertex
        self.verts_radius = np.ones(self.verts_coord.shape[0]) #* (self.RadiusEarth / self.SizeScale) #radius of each vertex's point (default = earth)
        self.verts_mass = np.ones(self.verts_coord.shape[0]) #* self.MassEarth #mass of each vertex

        self.parts_coord = self.generate_rand_coordinates(self.n_particles) #coordinates of vertices
        # #self.get_function_verts()#self.generate_rand_coordinates(self.n_particles) * self.sizescale * 10000 #coordinates of vertices
        self.parts_vel = np.zeros_like(self.parts_coord) #velocity of each vertex
        self.parts_radius = np.ones(self.parts_coord.shape[0]) #* (self.RadiusEarth / self.SizeScale) #radius of each vertex's point (default = earth)

        self.__initialize_coordinates__()

    def __initialize_coordinates__(self):
        self.verts_mass *= MassJupiter #jupiter
        self.verts_radius *= RadiusJupiter
        #Create a rotating color scheme so some spheres are red, others green and blue
        n = self.verts_coord.shape[0]
        self.verts_color = (np.mod(np.arange(n),3)[:,None] == np.arange(3)).astype(int)
        self.verts_color = np.c_[self.verts_color, np.ones((n,1))]

        self.parts_radius *= RadiusEarth
        self.parts_color = np.tile(np.array([1.0, 1.0, 1.0, 1.0]) * ColorEarth, (self.n_particles,1)).astype(np.float32)

    def generate_rand_coordinates(self, n):
        return (np.random.ranf(3 * n).reshape((n, 3)) - 0.5) * self.size_scale * 100

    def get_array_size(self):
        return self.verts_coord.shape[0] + self.parts_coord.shape[0]
