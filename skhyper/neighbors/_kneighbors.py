"""
K-nearest neighbors classification
"""
import operator
import functools
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as _sklearn_kneighbors

from skhyper.process import Process


class KNeighborsClassifier:
    """K-nearest neighbors classification

    Parameters
    ----------
    n_neighbors : int, default: 5
        Number of neighbors to use by default fro kneighbors queries

    weights : str or callable, default: 'uniform'
        Weight function used in prediction. Possible values:
            - 'uniform': uniform weights: All points in each neighborhood
                are weighted equally
            - 'distance': weight points by the inverse of their distance.
                In this case, closer neighbors of a query point will have
                a greater influence than neighbors which are further away.
            - [callable]: a user-defined function which accepts an array
                of distances, and returns an array of the same shape
                containing the weights.

    algorithm: str, default: 'auto'
         {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}
         Algorithm used to compute the nearest neighbors:
            - 'ball_tree' will use BallTree
            - 'kd_tree' will use KDTree
            - 'brute' will use a brute-force search.
            - 'auto' will attempt to decide the most appropriate algorithm
                based on the values passed to fit method.

    leaf_size : int, default: 30
        Leaf size passed to BallTree or KDTree. This can affect the speed
        of the construction and query, as well as the memory required to
        store the tree. The optimal value depends on the nature
        of the problem.

    p : int, default: 2
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and
        euclidean_distance (l2) for p = 2. For arbitrary p,
        minkowski_distance (l_p) is used.

    metric : str or callable, default: 'minkowski'
        the distance metric to use for the tree. The default metric
        is minkowski, and with p=2 is equivalent to the standard
        Euclidean metric. See the documentation of the DistanceMetric
        class for a list of available metrics.

    metric_params : dict, default: None
        Additional keyword arguments for the metric function

    n_jobs : int, default: 1
        The number of parallel jobs to run for the neighbors search. If
        ``-1``, then then number of jobs is set to the number of CPU cores.
        Doesn't affect the :class:`~skhyper.neighbors.KNeighborsClassifier.fit` method.

    """
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=1):
        self._X = None

        # sklearn KNeighborsClassifier model
        self.mdl = None

        # sklearn optional KNeighborsClassifier arguments
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def _check_is_fitted(self):
        """
        Performs a check to see if self.data is empty. If it is, then fit() has not been called yet.
        """
        if self._X is None:
            raise AttributeError('Data has not yet been fitted with fit()')

    def fit(self, X, y):
        """Fit the KNeighborsClassifier model according to the given training data.

        Parameters
        ----------
        X : object, Process instance

        y : array, shape (x, y, (z))
            True labels for X.

        Returns
        -------
        self : object
            Returns self.

        """
        self._X = X
        if not isinstance(self._X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        if type(y) != np.ndarray:
            raise TypeError('Target value array must be a numpy array')

        if len(y.shape) != 2 and len(y.shape) != 3:
            raise TypeError('Target value array must be 2- or 3-dimensional.')

        mdl = _sklearn_kneighbors(n_neighbors=self.n_neighbors, weights=self.weights,
                                  algorithm=self.algorithm, leaf_size=self.leaf_size,
                                  p=self.p, metric=self.metric, metric_params=self.metric_params,
                                  n_jobs=self.n_jobs)

        y_2d = np.reshape(y, functools.reduce(operator.mul, y.shape, 1))

        mdl.fit(self._X.flatten(), y_2d)

        self.mdl = mdl

        return self

    def predict(self, X):
        """Perform classification on samples in X.

        For an one-class model, +1 or -1 is returned.

        Parameters
        ----------
        X : object, Process instance

        Returns
        -------
        y_pred : array, shape (x, y, (z))
            Class labels for samples in X.

        """
        if not isinstance(X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        self._check_is_fitted()

        y_pred = self.mdl.predict(X.flatten())
        y_pred = np.reshape(y_pred, X.shape[:-1])

        return y_pred

    def predict_proba(self, X):
        """Compute probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute probability set to True.

        Parameters
        ----------
        X : object, Process instance

        Returns
        -------
        T : array, shape (x, y, (z), n_classes)
            Returns the probability of the sample for each class in the model.
            The columns correspond to the classes in sorted order, as they
            appear in the attribute classes_.

        """
        if not isinstance(X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        self._check_is_fitted()

        T = self.mdl.predict_proba(X.flatten())
        T = np.reshape(T, X.shape[:-1] + (T.shape[1],))

        return T

    def score(self, X, y, sample_weight=None):
        """Returns the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy which is
        a harsh metric since you require for each sample that each label s
        et be correctly predicted.

        Parameters
        ----------
        X : object, Process instance

        y : array, shape (x, y, (z))
            True labels for X.

        sample_weight : array, shape(x, y, (z)). Optional, default: None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of `self.predict(X)` wrt. y.

        """
        if not isinstance(X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        if type(y) != np.ndarray:
            raise TypeError('Target value array must be a numpy array')

        if len(y.shape) != 2 and len(y.shape) != 3:
            raise TypeError('Target value array must be 2- or 3-dimensional.')

        y_2d = np.reshape(y, functools.reduce(operator.mul, y.shape, 1))

        score = self.mdl.score(X.flatten(), y_2d, sample_weight=sample_weight)

        return score
