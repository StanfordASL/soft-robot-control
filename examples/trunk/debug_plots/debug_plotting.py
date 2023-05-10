import matplotlib.pyplot as plt
import pickle
import numpy as np
from os.path import dirname, abspath, join

path = abspath(dirname(__file__))

plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': 'FreeSerif'})
plt.rcParams.update({'mathtext.fontset': 'cm'})

FONTSCALE = 1.2

plt.rc('font', size=12*FONTSCALE)          # controls default text sizes
plt.rc('axes', titlesize=15*FONTSCALE)     # fontsize of the axes title
plt.rc('axes', labelsize=13*FONTSCALE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12*FONTSCALE)    # fontsize of the tick labels
plt.rc('legend', fontsize=8*FONTSCALE)    # legend fontsize
plt.rc('figure', titlesize=15*FONTSCALE)   # fontsize of the figure title
suptitlesize = 20*FONTSCALE

plt.rc('figure', autolayout=True)

def u():
    """plot the content of u.pkl"""
    with open(join(path, "u.pkl"), 'rb') as f:
        u = pickle.load(f)
    plt.plot(u)
    plt.suptitle(r"$u$")
    plt.show()

def x():
    """plot the content of x.pkl"""
    with open(join(path, "x.pkl"), 'rb') as f:
        x = pickle.load(f)
    # x = np.array(x)
    # print(x.shape)
    x = np.hstack([x[i] for i in range(len(x))]).T
    print(x.shape)
    plt.plot(x)
    plt.suptitle(r"$x$")
    plt.show()

def y():
    """plot the content of y.pkl"""
    with open(join(path, "y.pkl"), 'rb') as f:
        y = pickle.load(f)
    plt.plot(y)
    plt.suptitle(r"$y$")
    plt.show()

def ybar():
    """plot the content of ybar.pkl"""
    with open(join(path, "ybar.pkl"), 'rb') as f:
        ybar = pickle.load(f)
    plt.plot(ybar)
    plt.suptitle(r"$\bar{y}$")
    plt.show()

def xdot():
    """plot the content of xdot.pkl"""
    with open(join(path, "xdot.pkl"), 'rb') as f:
        xdot = pickle.load(f)
    plt.plot(xdot)
    plt.suptitle(r"$\dot{x}$")
    plt.show()


if __name__ == '__main__':
    # u()
    ybar()
    x()
    y()
    # xdot()