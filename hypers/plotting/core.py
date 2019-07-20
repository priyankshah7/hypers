import numpy as np
from typing import Union, Callable
import matplotlib.pyplot as plt

from hypers.plotting import attributes
if attributes.use_seaborn:
    import seaborn as sns
    sns.set()

def plot_array_spectrum(spcs: np.ndarray, title: str):
    def to_plot():
        a = plt.plot(spcs)
    _plotter(to_plot)


def plot_array_image(ims: np.ndarray):
    pass


def plot_learned_spectrum(n_components: Union[int, tuple],
                  spcs: np.ndarray):
    pass


def plot_learned_image(n_components: Union[int, tuple],
               ims: np.ndarray):
    pass


def plot_learned_components_grid(n_components: Union[int, tuple],
                         ims: np.ndarray,
                         spcs: np.ndarray):
    pass


def plot_learned_map_grid(lbls: np.ndarray, spcs: np.ndarray):
    pass


def _plotter(to_plot: Callable):
    plt.figure(figsize=attributes.figsize, facecolor=attributes.facecolor)
    to_plot()
    plt.legend()
    plt.show()
