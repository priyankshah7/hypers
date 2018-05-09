"""
Independent component analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA as _sklearn_ica

from skhyper.process import Process


class FastICA:
    """Independent component analysis (ICA)

    FastICA: a fast algorithm for Independent Component Analysis.

    Parameters
    ----------
    n_components : int or None
        Number of components to use. If none is passed, all are used.

    algorithm : {‘parallel’, ‘deflation’}
        Apply parallel or deflational algorithm for FastICA.

    whiten : boolean, optional
        If whiten is false, the data is already considered to be whitened, and
        no whitening is performed.

    fun : string or function, optional. Default: ‘logcosh’
        The functional form of the G function used in the approximation to
        neg-entropy. Could be either ‘logcosh’, ‘exp’, or ‘cube’. You can also
        provide your own function. It should return a tuple containing the
        value of the function, and of its derivative, in the point.
        Example::

            def my_g(x):
                return x ** 3, 3 * x ** 2`

    fun_args : dictionary, optional
        Arguments to send to the functional form. If empty and if fun=’logcosh’,
         fun_args will take value {‘alpha’ : 1.0}.

    max_iter : int, optional
        Maximum number of iterations during fit.

    tol : float, optional
        Tolerance on update at each iteration.

    w_init : None of an (n_components, n_components) ndarray
        The mixing matrix to be used to initialize the algorithm.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Attributes
    ----------
    image_components_ : array, shape (x, y, (z), n_features)

    spec_components_ : array, shape(n_features, n_features)

    mdl : object, FastICA() instance
        An instance of the FastICA model from scikit-learn used when
        instantiating FastICA() from scikit-hyper

    mixing_ : array, shape (n_features, n_components)
        The mixing matrix.

    n_iter_ : int
        If the algorithm is “deflation”, n_iter is the maximum number of
        iterations run across all components. Else they are just the number
        of iterations taken to converge.


    Examples
    --------
    >>> import numpy as np
    >>> from skhyper.process import Process
    >>> from skhyper.decomposition import FastICA
    >>>
    >>> test_data = np.random.rand(100, 100, 10, 1024)
    >>> X = Process(test_data)
    >>>
    >>> mdl = FastICA()
    >>> mdl.fit_transform(X)
    >>>
    >>> # To view the image/spectrum for each component:
    >>> # e.g.
    >>> import matplotlib.pyplot as plt
    >>>
    >>> plt.figure()
    >>> plt.subplot(121); mdl.image_components_[1]
    >>> plt.subplot(122); mdl.spec_components_[1]
    """
    def __init__(self, n_components=None, algorithm='parallel', whiten=True, fun='logcosh',
                 fun_args=None, max_iter=200, tol=1e-4, w_init=None, random_state=None):
        self._X = None

        # sklearn FastICA model
        self.mdl = None

        # custom decomposition attributes
        self.image_components_ = None
        self.spec_components_ = None

        # sklearn FastICA attributes
        self.mixing_ = None
        self.n_iter_ = None

        # sklearn optional FastICA arguments
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.fun_args = fun_args
        self.max_iter = max_iter
        self.tol = tol
        self.w_init = w_init
        self.random_state = random_state

    def _check_is_fitted(self):
        """
        Performs a check to see if self.data is empty. If it is, then fit_transform() has not been called yet.
        """
        if self._X is None:
            raise AttributeError('Data has not yet been fitted with fit_transform()')

    def fit_transform(self, X):
        """
        Fits the FastICA model to the processed hyperspectral array.

        Parameters
        ----------
        X : object, type (Process)

        Returns
        -------
        self : object
            Returns the instance itself

        """
        self._X = X
        if not isinstance(self._X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        mdl = _sklearn_ica(n_components=self.n_components, algorithm=self.algorithm, whiten=self.whiten,
                           fun=self.fun, fun_args=self.fun_args, max_iter=self.max_iter, tol=self.tol,
                           w_init=self.w_init, random_state=self.random_state)

        w_matrix = mdl.fit_transform(self._X.flatten())
        h_matrix = mdl.components_

        self.mdl = mdl
        self.mixing_ = mdl.mixing_
        self.n_iter_ = mdl.n_iter_

        self.image_components_ = np.reshape(w_matrix, self._X.shape)
        self.spec_components_ = h_matrix.T

        return self

    def inverse_transform(self, n_components):
        """
        Performs an inverse transform on the fitted data to project back to the original space.

        Parameters
        ----------
        n_components : int
            Number of components to keep when projected back to original space

        Returns
        -------
        Xd : object, Process instance of Xd
             Denoised hyperspectral data with the same dimensions as the array passed to fit_transform()

        """
        self._check_is_fitted()

        X_flat = self._X.flatten()

        mdl = _sklearn_ica(n_components=n_components, algorithm=self.algorithm, whiten=self.whiten,
                           fun=self.fun, fun_args=self.fun_args, max_iter=self.max_iter, tol=self.tol,
                           w_init=self.w_init, random_state=self.random_state)

        Xd_flat = mdl.fit_transform(X_flat)
        Xd_flat = mdl.inverse_transform(Xd_flat)

        Xd = np.reshape(Xd_flat, self._X.shape)

        return Process(Xd)
