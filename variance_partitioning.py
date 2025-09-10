"""
Visualization of suppression and facilitation in 2-regressor problems
@author: jdiedrichsen
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import PcmPy as pcm
import warnings
from numpy import sqrt, cos, sin,arccos, pi
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource

def calc_r2(r_y1, r_y2, r12):
    """
    Calculate the R^2 for the full model given the individual correlations
    and the correlation between the predictors.

    Args:
        r_y1 : float : Correlation of first predictor with the outcome
        r_y2 : float : Correlation of second predictor with the outcome
        r12 : float : Correlation between the two predictors

    Returns:
        RY12:  R^2 for the full model

    """
    RY12 = (2 * r_y1 * r_y2 * r12 - r_y1**2 - r_y2**2) / (r12**2 - 1)
    DET = 1 - r_y1**2 - r_y2**2 - r12**2 + 2 * r_y1 * r_y2 * r12
    return RY12, DET



def quadratic(a, b, c):
    """ Solution of quadratic equation"""
    D = b ** 2 - 4 * a * c
    x = (-b - sqrt(D)) / (2 * a)
    return x


def cubic(a, b, c, d):
    """ Solution of cubic equation"""
    if np.abs(a)<1e-10:
        return quadratic(b,c,d)
    solutions = []
    p = (3 * a * c - b ** 2) / (3 * a ** 2)
    q = (2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d) / (27 * a ** 3)
    solution = np.zeros(3)
    for n in range(3):
        solution[n]=((2 * sqrt(-p / 3) * cos(arccos((-3 * q) * sqrt(-3 * p) / (2 * p ** 2)) / 3 + 2 * pi * n / 3))
                        - (b / (3 * a)))
    if a < 0:
        return solution[0]
    else:
        return solution[2]

def define_surface():
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()

    a = 2*cos(v)*sin(u)*sin(v)*sin(v)*cos(u)
    l = np.empty((len(u),))

    for i in range(len(u)):
        l[i] = cubic(a[i],-1,0,1)
    pass
    RY1 = l*np.cos(u)*np.sin(v)
    RY2 = l*np.sin(u)*np.sin(v)
    R12 = l*np.cos(v)

    RY12 = (2*RY1*RY2*R12 - RY1**2 - RY2**2)/(R12**2-1)
    RY12[np.abs(R12)==1] = 0
    d = RY1**2+RY2**2-RY12 # Difference between combined and sum of single

    tri = mtri.Triangulation(u, v)
    xt = RY1[tri.triangles]
    yt = RY2[tri.triangles]
    zt = R12[tri.triangles]
    dt = d[tri.triangles].mean(axis=1)
    verts = np.stack((xt, yt, zt), axis=-1)
    return tri,verts,dt

def plot_surface(tri,verts,dt):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    cmap = plt.get_cmap('bwr')

    colors = cmap((dt+1))/2
    ls = LightSource(azdeg=10.0, altdeg=-90)
    polyc = Poly3DCollection(verts,shade=True,facecolors=colors,lightsource=ls)
    ax.add_collection3d(polyc)
    # uu = np.linspace(0, 2 * np.pi, 100)
    # ax.plot(cos(uu)*1.1,sin(uu)*1,np.zeros(uu.shape),color='k')
    ax.set_aspect('equal')
    ax.set_xticks([-1,-0.5,0,0.5,1])
    ax.set_yticks([-1,-0.5,0,0.5,1])
    ax.set_zticks([-1,-0.5,0,0.5,1])
    ax.set_xlim(-1.2,1.2)
    ax.set_ylim(-1.2,1.2)
    ax.set_zlim(-1.2,1.2)
    ax.view_init(20, 0,0)
    pass

def get_vectors(r12, rY1, rY2):
    X1 = np.array([1, 0, 0])
    X2 = np.array([r12, np.sqrt(1 - r12**2), 0])
    a = rY1
    b = (rY2 - rY1 * r12) / np.sqrt(1 - r12**2)
    c = np.sqrt(1 - a**2 - b**2)
    Y = np.array([a, b, c])
    return X1, X2, Y


def plot_projection(r12, rY1, rY2,ax = None):
    if ax is None:
        ax = plt.figure().add_subplot(projection='3d')

    X1, X2, Y = get_vectors(r12, rY1, rY2)

    ax.quiver(0, 0, 0, Y[0], Y[1], Y[2], length=1, arrow_length_ratio=0.05, colors='k')
    ax.quiver(0, 0, 0, X1[0], X1[1], X1[2], length=1, arrow_length_ratio=0.05, colors='k')
    ax.quiver(0, 0, 0, X2[0], X2[1], X2[2], length=1, arrow_length_ratio=0.05, colors='k')

    P1 = X1 * (X1.T @ Y)
    P2 = X2 * (X2.T @ Y)
    X = np.c_[X1,X2]
    P = X @ np.linalg.pinv(X) @ Y
    Q = P1 + P2
    # the line projected onto <x1, X2> plane
    ax.plot([0, P1[0]], [0, P1[1]], [0,P1[2]], linestyle='-', color='g', linewidth=1.5)
    ax.plot([0, P2[0]], [0, P2[1]], [0,P2[2]], linestyle='-', color='g', linewidth=1.5)
    ax.plot([0, P[0]], [0, P[1]], [0,P[2]], linestyle='-', color='g', linewidth=1.5)
    ax.plot([P1[0], Q[0]], [P1[1],Q[1]], [P1[2],Q[2]], linestyle='--', color='r', linewidth=1)
    ax.plot([P2[0], Q[0]], [P2[1],Q[1]], [P2[2],Q[2]], linestyle='--', color='r', linewidth=1)

    ax.plot([Y[0], P1[0]], [Y[1],P1[1]], [Y[2],P1[2]], linestyle='--', color='b', linewidth=1)
    ax.plot([Y[0], P2[0]], [Y[1],P2[1]], [Y[2],P2[2]], linestyle='--', color='b', linewidth=1)
    ax.plot([Y[0], P[0]], [Y[1],P[1]], [Y[2],P[2]], linestyle='--', color='b', linewidth=1)

    ax.set_xlim([0,1.05])
    ax.set_ylim([0,1.05])
    ax.set_zlim([0,1.05])

    R2,det = calc_r2(rY1,rY2,r12)
    print(det)
    title_str= f'{r12} {rY1} {rY2} {R2:2.2f} vs. {rY1**2 + rY2**2:2.2f} '
    ax.set_title(title_str)


if __name__=="__main__":
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(1,3,1,projection='3d')
    plot_projection(0,0.6,0.45,ax1)
    ax2 = fig.add_subplot(1,3,2,projection='3d')
    plot_projection(0.7,0.6,0.6,ax2)
    ax3 = fig.add_subplot(1,3,3,projection='3d')
    plot_projection(0.7,0.2,0.6,ax3)


    pass