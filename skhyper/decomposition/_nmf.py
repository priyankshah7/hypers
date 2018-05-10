"""
Non-negative matrix factorization
"""
import numpy as np
from sklearn.decomposition import NMF as _sklearn_nmf

from skhyper.process import Process


class NMF:
    """Non-negative matrix factorization

    Find two non-negative matrices (W, H) whose product approximates the
    non-negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    Parameters
    ----------
    n_components : int or None
        Number of components to use. If none is passed, all are used.

    init : string, 'random' | 'nndsvd' | 'nndsvda' | 'nndsvdar' | 'custom'
        Method used to initialize the procedure. Default: ‘nndsvd’ if
        n_components < n_features, otherwise random. Valid options:

        - ‘random’: non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - ‘nndsvd’: Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - ‘nndsvda’: NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - ‘nndsvdar’: NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - ‘custom’: use custom matrices W and H

    solver : string, ‘cd’ | ‘mu’
        Numerical solver to use: ‘cd’ is a Coordinate Descent solver.
        ‘mu’ is a Multiplicative Update solver.

    beta_loss : float or string, default 'frobenius'
        String must be in {‘frobenius’, ‘kullback-leibler’, ‘itakura-saito’}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from ‘frobenius’
        (or 2) and ‘kullback-leibler’ (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or ‘itakura-saito’), the input
        matrix X cannot contain zeros. Used only in ‘mu’ solver.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : int, default: 200
        Maximum number of iterations before timing out.

    random_state : int, RandomState instance or None. default: None
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by np.random.

    alpha : double, default: 0
        Constant that multiplies the regularization terms. Set it
        to zero to have no regularization.

    l1_ratio : double, default: 0
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm). For l1_ratio = 1 it is an elementwise L1
        penalty. For 0 < l1_ratio < 1, the penalty is a combination of
        L1 and L2.

    verbose : bool, default: False
        Whether to be verbose.

    shuffle : bool, default: False
        If true, randomize the order of coordinates in the CD solver.

    Attributes
    ----------
    image_components_ : array, shape (x, y, (z), n_features)

    spec_components_ : array, shape(n_features, n_features)

    mdl : object, NMF() instance
        An instance of the NMF model from scikit-learn used when
        instantiating NMF() from scikit-hyper

    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence,
        between the training data X and the reconstructed data WH
        from the fitted model.

    n_iter_ : int
        Actual number of iterations.


    Examples
    --------
    >>> import numpy as np
    >>> from skhyper.process import Process
    >>> from skhyper.decomposition import NMF
    >>>
    >>> test_data = np.random.rand(100, 100, 10, 1024)
    >>> X = Process(test_data)
    >>>
    >>> mdl = NMF()
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
    def __init__(self, n_components=None, init=None, solver='cd', beta_loss='frobenius', tol=0.0001,
                 max_iter=200, random_state=None, alpha=0.0, l1_ratio=0.0, verbose=0, shuffle=False):
        self._X = None

        # sklearn NMF model
        self.mdl = None

        # custom decomposition attributes
        self.image_components_ = None
        self.spec_components_ = None

        # sklearn NMF attributes
        self.reconstruction_err_ = None
        self.n_iter_ = None

        # sklearn optional NMF arguments
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle

    def _check_is_fitted(self):
        """
        Performs a check to see if self.data is empty. If it is, then fit_transform() has not been called yet.
        """
        if self._X is None:
            raise AttributeError('Data has not yet been fitted with fit_transform()')

    def fit_transform(self, X):
        """
        Fits the NMF model to the processed hyperspectral array.

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

        mdl = _sklearn_nmf(n_components=self.n_components, init=self.init, solver=self.solver,
                           beta_loss=self.beta_loss, tol=self.tol, max_iter=self.max_iter,
                           random_state=self.random_state, alpha=self.alpha, l1_ratio=self.l1_ratio,
                           verbose=self.verbose, shuffle=self.shuffle)

        w_matrix = mdl.fit_transform(self._X.flatten())
        h_matrix = mdl.components_

        self.mdl = mdl
        self.reconstruction_err_ = mdl.reconstruction_err_
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

        mdl = _sklearn_nmf(n_components=n_components, init=self.init, solver=self.solver,
                           beta_loss=self.beta_loss, tol=self.tol, max_iter=self.max_iter,
                           random_state=self.random_state, alpha=self.alpha, l1_ratio=self.l1_ratio,
                           verbose=self.verbose, shuffle=self.shuffle)

        Xd_flat = mdl.fit_transform(X_flat)
        Xd_flat = mdl.inverse_transform(Xd_flat)

        Xd = np.reshape(Xd_flat, self._X.shape)

        return Process(Xd)

