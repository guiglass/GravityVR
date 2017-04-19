import numpy as np
from .rotation_matrix import rotation_matrix
from .orbital_velocity import get_orbital_velocity

def make_circula_pts(n_particles, axis, theta):

    y = np.linspace(0, (2 * np.pi), n_particles+1)[:-1]

    p = np.zeros((n_particles, 3))
    pts = np.dstack([np.sin(y), np.cos(y), np.zeros(n_particles)])[0]

    for i, pt in enumerate(pts):
        s = np.dot(rotation_matrix(axis, theta), pt)
        p[i] = s

    verts = np.zeros((n_particles, 3))

    verts[:,0] = np.sin(y)
    verts[:,1] = np.cos(y)
    verts[:,2] = np.zeros(n_particles)

    for i, pt in enumerate(verts):
         verts[i] = np.dot(rotation_matrix(axis, theta), pt)

    return verts, pts

def get_rings(radius, center_coord, center_vel, center_mass, n_particles=20, particle_size=1*10**4, particle_color=(1.0, 1.0, 1.0, 1.0), axis=(0,0,1), theta=0):

    verts_vel = np.zeros((n_particles, 3))
    verts_radius = np.ones(n_particles) * particle_size  # size of point
    verts_color = np.ones((n_particles, 4)) * np.array(particle_color)

    verts_rotated, verts = make_circula_pts(n_particles, axis, theta)# Create ring system then rotate verts using euler rodrigues transformation
    verts_coord = verts_rotated * radius   # coordinates of vertices

    for i, pt in enumerate(verts * radius):#use the ring and calc the axial velocity, then apply euler rodrigues transformation to the velocities so as to match the tilted ring
        v = get_orbital_velocity((0,0,0), center_mass, pt)
        v = np.dot(rotation_matrix(axis, theta), v)

        verts_vel[i] = v - center_vel

    return verts_coord+center_coord, verts_vel, verts_radius, verts_color