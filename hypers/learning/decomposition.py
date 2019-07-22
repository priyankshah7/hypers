import numpy as np
import hypers as hp
from typing import Tuple, Union, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FastICA, NMF

sns.set()


class decompose:
    """ Provides instances of decomposition classes """
    def __init__(self, X: 'hp.hparray'):
        self.pca = pca(X)
        self.ica = ica(X)
        self.nmf = nmf(X)
        self.vca = vca(X)


class pca:
    """
    Principal components analysis

    Parameters
    ----------
    X: hp.hparray
        The hyperspectral array.

    Attributes
    ----------
    ims: np.ndarray
        Images of the principal components

    spcs: np.ndarray
        Spectra of the principal components
    """
    def __init__(self, X: 'hp.hparray'):
        self.X = X
        self.ims = None
        self.spcs = None
        self._mdl = None

    def scree(self):
        """
        Calculated scree

        Returns an array which assigns the amount of variance contributed by each principal
        component for the dataset.

        Returns
        -------
        np.ndarray
            PCA scree
        """
        mdl = PCA()
        mdl.fit_transform(self.X.collapse())

        return mdl.explained_variance_ratio_

    def plot_scree(self):
        """
        Scree plot

        Calculates and returns a scree plot for the dataset
        """
        mdl = PCA()
        mdl.fit_transform(self.X.collapse())

        plt.plot(mdl.explained_variance_ratio_)
        plt.xlabel('Principal components')
        plt.ylabel('Variance ratio')
        plt.title('Scree plot')
        plt.tight_layout()
        plt.show()

    def calculate(self, n_components: int=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate principal components

        Parameters
        ----------
        n_components: int
            The number of components to calculate. This should be less than or equal to the size
            of the spectral dimension.

        kwargs:
            Keyword arguments can be supplied for scikit-learn's PCA class.
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

        Returns
        -------
        np.ndarray:
            Images of the first n_components number of principal components. The shape of the array will
            be (nspatial, n_components).

        np.ndarray:
            Spectra of the first n_components number of principal components. The shape of the array will
            be (nfeatures, n_components).
        """
        if n_components is None:
            n_components = self.X.shape[-1]

        self._mdl = PCA(n_components=n_components, **kwargs)
        self.ims = self._mdl.fit_transform(self.X.collapse()).reshape(self.X.data.shape[:-1] + (n_components,))
        self.spcs = self._mdl.components_.transpose()

        return self.ims, self.spcs

    def reduce(self, n_components: int=None):
        if self._mdl is None:
            _, _ = self.calculate(n_components=n_components)

        if n_components is None:
            n_components = self.ims.shape[-1]

        ims = self.ims.reshape(np.prod(self.ims.shape[:-1]), self.ims.shape[-1])
        inversed = self._mdl.inverse_transform(ims[..., :n_components])
        return inversed.reshape(self.ims.shape[:-1] + (self.spcs.shape[0],))

    def plot_image(self, n_components: Union[int, Tuple] = 1):
        _plot_image(n_components=n_components, ims=self.ims)

    def plot_spectrum(self, n_components: Union[int, Tuple] = 1):
        _plot_spectrum(n_components=n_components, spcs=self.spcs)

    def plot_grid(self, n_components: Union[int, Tuple] = 1):
        _plot_grid(n_components=n_components, ims=self.ims, spcs=self.spcs)


class ica:
    def __init__(self, X: 'hp.hparray'):
        self.X = X
        self.ims = None
        self.spcs = None

    def calculate(self, n_components: int = 4, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if n_components is None:
            n_components = self.X.shape[-1]

        mdl = FastICA(n_components=n_components, **kwargs)
        self.ims = mdl.fit_transform(self.X.collapse()).reshape(self.X.data.shape[:-1] + (n_components,))
        self.spcs = mdl.components_.transpose()

        return self.ims, self.spcs

    def plot_image(self, n_components: Union[int, Tuple] = 1):
        _plot_image(n_components=n_components, ims=self.ims)

    def plot_spectrum(self, n_components: Union[int, Tuple] = 1):
        _plot_spectrum(n_components=n_components, spcs=self.spcs)

    def plot_grid(self, n_components: Union[int, Tuple] = 1):
        _plot_grid(n_components=n_components, ims=self.ims, spcs=self.spcs)


class nmf:
    def __init__(self, X: 'hp.hparray'):
        self.X = X
        self.ims = None
        self.spcs = None

    def calculate(self, n_components: int = 4, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if n_components is None:
            n_components = self.X.shape[-1]

        mdl = NMF(n_components=n_components, **kwargs)
        self.ims = mdl.fit_transform(self.X.collapse()).reshape(self.X.data.shape[:-1] + (n_components,))
        self.spcs = mdl.components_.transpose()

        return self.ims, self.spcs

    def plot_image(self, n_components: Union[int, Tuple] = 1):
        _plot_image(n_components=n_components, ims=self.ims)

    def plot_spectrum(self, n_components: Union[int, Tuple] = 1):
        _plot_spectrum(n_components=n_components, spcs=self.spcs)

    def plot_grid(self, n_components: Union[int, Tuple] = 1):
        _plot_grid(n_components=n_components, ims=self.ims, spcs=self.spcs)


class vca:
    def __init__(self, X: 'hp.hparray'):
        self.X = X
        self.n_components = None
        self.spcs = None
        self.coords = None

    def calculate(self, n_components: int = 4,
                  input_snr: float = 0) -> Tuple[np.ndarray, List[int]]:

        self.n_components = n_components
        Ae, indice, Yp = self._calcluate_vca(self.X.collapse().T, n_components, snr_input=input_snr)
        index = [0] * n_components
        for component in range(n_components):
            index[component] = np.unravel_index(indice[component], self.X.shape[:-1])

        self.spcs = Ae
        self.coords = index

        return Ae, index

    def plot_spectrum(self, n_components: Union[int, Tuple] = None):
        if n_components is None:
            n_components = self.n_components

        if type(n_components) is int:
            for component in range(n_components):
                plt.plot(self.spcs[:, component], label='Comp. ' + str(component + 1))
            plt.tight_layout()
            plt.show()

        elif type(n_components) is tuple and len(n_components) == 2:
            pass

        else:
            raise ValueError('n_components must either be an integer or a tuple of 2 values')

    @staticmethod
    def _estimate_snr(Y: np.ndarray,
                      r_m: np.ndarray,
                      x: np.ndarray) -> np.ndarray:
        [L, N] = Y.shape  # L number of bands (channels), N number of pixels
        [p, N] = x.shape  # p number of endmembers (reduced dimension)

        P_y = np.sum(Y ** 2) / float(N)
        P_x = np.sum(x ** 2) / float(N) + np.sum(r_m ** 2)
        snr_est = 10 * np.log10((P_x - p / L * P_y) / (P_y - P_x))

        return snr_est

    def _calcluate_vca(self, Y, R, verbose=True, snr_input=0.0):
        # Vertex Component Analysis
        #
        # Ae, indice, Yp = vca(Y,R,verbose = True,snr_input = 0)
        #
        # ------- Input variables -------------
        #  Y - matrix with dimensions L(channels) x N(pixels)
        #      each pixel is a linear mixture of R endmembers
        #      signatures Y = M x s, where s = gamma x alfa
        #      gamma is a illumination perturbation factor and
        #      alfa are the abundance fractions of each endmember.
        #  R - positive integer number of endmembers in the scene
        #
        # ------- Output variables -----------
        # Ae     - estimated mixing matrix (endmembers signatures)
        # indice - pixels that were chosen to be the most pure
        # Yp     - Data matrix Y projected.
        #
        # ------- Optional parameters---------
        # snr_input - (float) signal to noise ratio (dB)
        # v         - [True | False]
        # ------------------------------------
        #
        # Author: Adrien Lagrange (adrien.lagrange@enseeiht.fr)
        # This code is a translation of a matlab code provided by
        # Jose Nascimento (zen@isel.pt) and Jose Bioucas Dias (bioucas@lx.it.pt)
        # available at http://www.lx.it.pt/~bioucas/code.htm under a non-specified Copyright (c)
        # Translation of last version at 22-February-2018 (Matlab version 2.1 (7-May-2004))
        #
        # more details on:
        # Jose M. P. Nascimento and Jose M. B. Dias
        # "Vertex Component Analysis: A Fast Algorithm to Unmix Hyperspectral Data"
        # submited to IEEE Trans. Geosci. Remote Sensing, vol. .., no. .., pp. .-., 2004
        #
        #

        #############################################
        # Initializations
        #############################################
        if len(Y.shape) != 2:
            raise ValueError('Input data must be of size L (number of bands i.e. channels) by N (number of pixels)')

        [L, N] = Y.shape  # L number of bands (channels), N number of pixels

        R = int(R)
        if R < 0 or R > L:
            raise ValueError('ENDMEMBER parameter must be integer between 1 and L')

        #############################################
        # SNR Estimates
        #############################################

        if snr_input == 0:
            y_m = np.mean(Y, axis=1, keepdims=True)
            Y_o = Y - y_m  # data with zero-mean
            Ud = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :R]  # computes the R-projection matrix
            x_p = np.dot(Ud.T, Y_o)  # project the zero-mean data onto p-subspace

            SNR = self._estimate_snr(Y, y_m, x_p);

            if verbose:
                print("SNR estimated = {}[dB]".format(SNR))
        else:
            SNR = snr_input
            if verbose:
                print("input SNR = {}[dB]\n".format(SNR))

        SNR_th = 15 + 10 * np.log10(R)

        #############################################
        # Choosing Projective Projection or
        #          projection to p-1 subspace
        #############################################

        if SNR < SNR_th:
            if verbose:
                print("... Select proj. to R-1")

                d = R - 1
                if snr_input == 0:  # it means that the projection is already computed
                    Ud = Ud[:, :d]
                else:
                    y_m = np.mean(Y, axis=1, keepdims=True)
                    Y_o = Y - y_m  # data with zero-mean

                    Ud = np.linalg.svd(np.dot(Y_o, Y_o.T) / float(N))[0][:, :d]  # computes the p-projection matrix
                    x_p = np.dot(Ud.T, Y_o)  # project thezeros mean data onto p-subspace

                Yp = np.dot(Ud, x_p[:d, :]) + y_m  # again in dimension L

                x = x_p[:d, :]  # x_p =  Ud.T * Y_o is on a R-dim subspace
                c = np.amax(np.sum(x ** 2, axis=0)) ** 0.5
                y = np.vstack((x, c * np.ones((1, N))))
        else:
            if verbose:
                print("... Select the projective proj.")

            d = R
            Ud = np.linalg.svd(np.dot(Y, Y.T) / float(N))[0][:, :d]  # computes the p-projection matrix

            x_p = np.dot(Ud.T, Y)
            Yp = np.dot(Ud, x_p[:d, :])  # again in dimension L (note that x_p has no null mean)

            x = np.dot(Ud.T, Y)
            u = np.mean(x, axis=1, keepdims=True)  # equivalent to  u = Ud.T * r_m
            y = x / np.dot(u.T, x)

        #############################################
        # VCA algorithm
        #############################################

        indice = np.zeros((R), dtype=int)
        A = np.zeros((R, R))
        A[-1, 0] = 1

        for i in range(R):
            w = np.random.rand(R, 1);
            f = w - np.dot(A, np.dot(np.linalg.pinv(A), w))
            f = f / np.linalg.norm(f)

            v = np.dot(f.T, y)

            indice[i] = np.argmax(np.absolute(v))
            A[:, i] = y[:, indice[i]]  # same as x(:,indice(i))

        Ae = Yp[:, indice]

        return Ae, indice, Yp


def _plot_spectrum(n_components: Union[int, Tuple],
                   spcs: np.ndarray):
    if spcs is None:
        raise ValueError('The calculate method must be called prior to plotting')

    if type(n_components) is int:
        plt.plot(spcs[..., n_components - 1], label='Comp. ' + str(n_components))
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif type(n_components) is tuple and len(n_components) == 2:
        for _comp in range(n_components[0], n_components[1]):
            plt.plot(spcs[..., _comp - 1], label='Comp. ' + str(_comp))
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError('n_components must either be an integer or a tuple of 2 values')


def _plot_image(n_components: Union[int, Tuple],
                ims: np.ndarray):
    if ims is None:
        raise ValueError('The calculate method must be called prior to plotting')

    if type(n_components) is int:
        if ims.ndim == 3:
            plt.imshow(np.squeeze(ims[..., n_components - 1]))

        elif ims.ndim == 4:
            plt.imshow(np.squeeze(np.mean(ims[..., n_components - 1], 2)))
        plt.title('Comp. ' + str(n_components))
        plt.tight_layout()
        plt.show()

    elif type(n_components) is tuple and len(n_components) == 2:
        _rows_total = ((n_components[1] - n_components[0]) // 4) + 1

        for _comp in range(n_components[0], n_components[1]):
            plt.subplot(_rows_total, 4, (_comp - n_components[0]) + 1)
            if ims.ndim == 3:
                plt.imshow(np.squeeze(ims[..., _comp - 1]))

            elif ims.ndim == 4:
                plt.imshow(np.squeeze(np.mean(ims[..., _comp - 1], 2)))
            plt.title('Comp. ' + str(_comp))
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError('n_components must either be an integer or a tuple of 2 values')


def _plot_grid(n_components: Union[int, Tuple],
               ims: np.ndarray,
               spcs: np.ndarray):
    if ims is None:
        raise ValueError('The calculate method must be called prior to plotting')

    if type(n_components) is int:
        plt.subplot(121)
        if ims.ndim == 3:
            plt.imshow(np.squeeze(ims[..., n_components - 1]))

        elif ims.ndim == 4:
            plt.imshow(np.squeeze(np.mean(ims[..., n_components - 1], 2)))

        plt.subplot(122)
        plt.plot(spcs[..., n_components - 1], label='Comp. ' + str(n_components))
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif type(n_components) is tuple and len(n_components) == 2:
        _comp_total = n_components[1] - n_components[0]

        for _comp in range(n_components[0], n_components[1]):
            plt.subplot(_comp_total, 2, 2 * (_comp - n_components[0]) + 1)
            if ims.ndim == 3:
                plt.imshow(np.squeeze(ims[..., _comp - 1]))

            elif ims.ndim == 4:
                plt.imshow(np.squeeze(np.mean(ims[..., _comp - 1], 2)))
            plt.title('Comp. ' + str(_comp))

            plt.subplot(_comp_total, 2, 2 * (_comp - n_components[0]) + 2)
            plt.plot(spcs[..., _comp - 1], label='Comp. ' + str(_comp))

        plt.tight_layout()
        plt.show()
