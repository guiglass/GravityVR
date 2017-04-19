import numpy as np

def get_orbital_velocity(pt1, pt1_mass, pt2, G=6.674*10**-11):

    X_Len = pt2[0] - pt1[0]
    Y_Len = pt2[1] - pt1[1]
    Z_Len = pt2[2] - pt1[2]

    B1 = np.sqrt(X_Len ** 2 + Y_Len ** 2)
    R = np.sqrt(B1 ** 2 + Z_Len ** 2)

    theta1 = np.arctan2(Y_Len, X_Len)

    theta2 = np.arctan2(Z_Len, B1)

    m = pt1_mass
    ag = G * m / R ** 2  # the gravitational acceleration at altitude

    # centripetal acceleration of the of the orbiting object = V**2/R
    V = np.sqrt(R * ag)

    ax1_v = V * -np.sin(theta1)
    ax2_v = V * np.cos(theta1)
    ax3_v = V * np.sin(theta2)

    return np.array([ax1_v, ax2_v, ax3_v])