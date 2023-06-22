from matplotlib import pyplot as plt
from frequency_domain.function_builders import generate_discrete_field
import numpy as np


def get_callbacks():
    def plot_results(P, *args, **kwargs):
        P = P.real
        plt.imshow(P)
        plt.show()
    return {
        'on_after_solve_2d_helmholtz': [plot_results,],
    }


def get_fwi_step_callbacks():
    def plot_fwi_step(J, mu, phi_new, *args, **kwargs):
        plt.subplot(1, 2, 1)
        plt.imshow(phi_new)
        plt.subplot(1, 2, 2)
        plt.imshow(phi_new > 0)
        plt.show()
    return {
        'on_after_step': [plot_fwi_step,],
    }


def plot_case(case, sources=None, receivers=None, show=True):
    """
    Helper function to plot forward case mesh properties mu and eta.
    Alternatively, sources and receiver positions can be given, just to show
    their positions in the plot.
    """
    mesh = case.mesh
    mu_field = generate_discrete_field(mesh, case.mu_function)
    eta_field = generate_discrete_field(mesh, case.eta_function)
    mean_points = np.mean(mesh.points_in_elements, axis=1)
    xs = mean_points[0:mesh.ny,0]
    ys = mean_points[::mesh.nx,1]

    plt.pcolormesh(xs, ys, mu_field, cmap='Blues')
    plt.pcolormesh(xs, ys, eta_field, cmap='Reds', alpha=0.05)
    if sources is not None and len(sources) > 0:
        plt.plot(sources[:, 0], sources[:, 1], 'ro')
    if receivers is not None and len(receivers) > 0:
        plt.plot(receivers[:, 0], receivers[:, 1], 'gx')

    if show:
        plt.show()


def plot_forward_results(case, results, ncols=2, show=True):
    """
    Helper function to plot forward problem results.
    """
    from math import ceil
    fig, ax = plt.subplots(ceil(len(results) / ncols), ncols)
    ax = ax.flatten()
    xs = case.mesh.points[0:case.mesh.nx + 1, 0]
    ys = case.mesh.points[::case.mesh.nx + 1, 1]
    for i, P in enumerate(results):
        ax[i].set_xlabel("km")
        ax[i].set_ylabel("km")
        im = ax[i].pcolormesh(xs, ys, P, cmap='magma')

    plt.subplots_adjust(wspace=0.2, right=0.75, top=0.95, bottom=0.05)
    cbar_ax = fig.add_axes([0.80, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    if show:
        plt.show()
