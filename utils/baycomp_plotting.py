"""
Functions for plotting baycomp's posterior plots.

Dependencies:
pip3 install iteround

``tern``
    A ternary plot.
``dens``
    A density plot.
Author: Mario Juez-Gil <mariojg@ubu.es>
"""
import types
import numpy as np
import matplotlib
import matplotlib.patches as patches
import matplotlib.colors as clrs
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.colors import ListedColormap
from math import sqrt, sin, cos, pi
from iteround import saferound
from scipy.interpolate import interpn
from scipy import stats

__all__ = ['Color', 'tern', 'dens']

# Custom colormap -> dark blue means high density, and light blue low
BLUES_CMAP = np.ones((256, 4))
BLUES_CMAP[:, 0] = np.linspace(130/256, 58/256, 256)
BLUES_CMAP[:, 1] = np.linspace(190/256, 111/256, 256)
BLUES_CMAP[:, 2] = np.linspace(176/256, 99/256, 256)
BLUES_CMAP = ListedColormap(BLUES_CMAP)

# Custom colors
class Color():
    BLUE = clrs.to_hex((130/255, 190/255, 176/255))
    GRAY = clrs.to_hex((77/255, 80/255, 94/255))
    BORDEAUX = clrs.to_hex((216/255, 149/255, 91/255))
    GREEN = clrs.to_hex((5/255, 126/255, 121/255))

def project(pts):
    SQRT_3 = sqrt(3)
    PI_6 = pi / 6

    p1, p2, p3 = pts.T / SQRT_3
    x = (p2 - p1) * cos(PI_6) + .5
    y = p3 - (p1 + p2) * sin(PI_6) + 1 / (2 * SQRT_3)

    return np.vstack((x, y)).T

def tern(p, l_tag = "L", r_tag="R", label_best_color = "#d8955b"):
    plt.style.use('classic')

    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.set_xlim(-.1, 1.1)
    ax.set_ylim(-.1, 1.1)



    # inner lines
    cx, cy = project(np.array([[1/3, 1/3, 1/3]]))[0] # central point
    for x, y in project(np.array([[.5, .5, 0], [.5, 0, .5], [0, .5, .5]])):
        ax.add_line(Line2D([cx, x], [cy, y], color='k', lw=2))

    # outer border of the triangle
    vert_coords = [(0., 0.), (.5, sqrt(3)/2), (1., 0.), (0., 0.)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    triangle = Path(vert_coords, codes)
    patch = patches.PathPatch(triangle, facecolor='none', lw=3)
    ax.add_patch(patch)

    # draw points
    tripts = project(p.sample[:, [0, 2, 1]])
    data, xe, ye = np.histogram2d(tripts[:, 0], tripts[:, 1], bins=30)
    z = interpn((.5 * (xe[1:] + xe[:-1]), .5 * (ye[1:] + ye[:-1])), data, 
            np.vstack([tripts[:, 0], tripts[:, 1]]).T, method='splinef2d', 
            bounds_error=False)
    idx = z.argsort()
    ax.scatter(tripts[:, 0][idx], tripts[:, 1][idx], c=z[idx], clip_path=patch,
            s=50, linewidth=0, cmap=BLUES_CMAP, rasterized=True)

    # vertices texts
    probs = saferound(list(p.probs()), places=4)

    # L
    #l_tag = '$\mathrm{' + l_tag + '}$'
    l_color = label_best_color if probs[0] == max(probs) else "black"

    ax.text(-.04, -.02, l_tag, ha='center', va='top', fontsize=30, color=l_color)
    ax.text(-.04, -.12, r'$\mathbf{(%.4f)}$'%probs[0], ha='center', va='top', fontsize=32, color=l_color)

    # ROPE
    rope_color = label_best_color if probs[1] == max(probs) else "black"

    ax.text(.5, 1, r'$\mathrm{ROPE}$', ha='center', va='bottom', fontsize=30, color=rope_color)
    ax.text(.5, .87, r'$\mathbf{(%.4f)}$'%probs[1], ha='center', va='bottom', fontsize=32, color = rope_color)

    # R
    #r_tag = '$\mathrm{' + r_tag + '}$'
    r_color = label_best_color if probs[2] == max(probs) else "black"

    ax.text(1.04, -.02, r_tag, ha='center', va='top', fontsize=30, color=r_color)
    ax.text(1.04, -.12, r'$\mathbf{(%.4f)}$'%probs[2], ha='center', va='top', fontsize=32, color=r_color)



    fig.tight_layout()
    return fig

def dens(p, label, ls='-', color=Color.BLUE):
    def add_posterior(ax, p, label, ls, color):
        def _update_yticks():
            tick = ax.max_y / 3
            yticks = [0, tick*1, tick*2, tick*3]
            ax.set_ylim(0, ax.max_y)
            ax.set_yticks(yticks)
            ax.set_yticklabels([r'$%.3f$'%round(x, 3) for x in yticks])

        targs = (p.df, p.mean, np.sqrt(p.var))
        x = np.linspace(min(stats.t.ppf(.005, *targs), -1.05 * p.rope),
                max(stats.t.ppf(.995, *targs), 1.05 * p.rope), 100)
        y = stats.t.pdf(x, *targs)
        y = y / y.sum() # density
        ax.plot(x, y, c=color, linestyle=ls, linewidth=2, 
                label=r'$\mathrm{%s}$'%label, zorder=ax.zo)
        ax.fill_between(x, y, facecolor=color, alpha=.1, edgecolor='none', 
                zorder=ax.zo) 
        ax.zo = ax.zo - 1

        if(ax.max_y == None):
            ax.max_y = np.amax(y) + np.amax(y)*.02
            _update_yticks()
        else:
            curr_max_y = np.amax(y) + np.amax(y)*.02
            if(curr_max_y > ax.max_y):
                ax.max_y = curr_max_y
                _update_yticks()

    plt.style.use('classic')
    plt.rc('legend', fontsize=25)

    # figure customization
    fig, ax = plt.subplots()
    fig.patch.set_alpha(0)
    ax.axvline(.01, c='darkorange', linewidth=2, zorder=101)
    ax.axvline(-.01, c='darkorange', linewidth=2, zorder=101)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('none')
    ax.tick_params(axis='both', which='major', labelsize=30, direction='inout',
            width=1, length=5)
    ax.set_xticks([])
    
    # appending the posterior to the axes
    ax.max_y = None
    ax.zo = 100
    add_posterior(ax, p, label, ls, color)
    fig.add_posterior = types.MethodType(add_posterior, ax)
    
    fig.tight_layout()
    return fig
