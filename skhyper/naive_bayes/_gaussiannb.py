"""
Gaussian Naive Bayes
"""
import operator
import functools
import numpy as np
from sklearn.naive_bayes import GaussianNB as _sklearn_gaussiannb

from skhyper.process import Process


class GaussianNB:
    """Gaussian Naive Bayes

    Implements the Gaussian Naive Bayes algorithm for classification.
    The likelihood of the features is assumed to be Gaussian.

    Parameters
    ----------
    priors : array, shape (n_classes)
        Prior probabilities of the classes. If specified, the priors are not
        adjusted according to the data.

    Attributes
    ----------
    class_prior_ : array, shape (n_classes)
        Probability of each class.

    class_count_ : array, shape (n_classes)
        Number of training samples observed in each class.

    theta_ : array, shape (n_classes, n_features)
        Mean of each feature per class

    sigma_ : array, shape (n_classes, n_features)
        Variance of each feature per class
    """
    def __init__(self, priors=None):
        self._X = None

        # sklearn GaussianNB model
        self.mdl = None

        # sklearn GaussianNB attributes
        self.class_prior_ = None
        self.class_count_ = None
        self.theta_ = None
        self.sigma_ = None

        # sklearn optional GaussianNB arguments
        self.priors = priors

    def _check_is_fitted(self):
        """
        Performs a check to see if self.data is empty. If it is, then fit() has not been called yet.
        """
        if self._X is None:
            raise AttributeError('Data has not yet been fitted with fit()')

    def fit(self, X, y, sample_weight=None):
        """Fit the SVM model according to the given training data.

        Parameters
        ----------
        X : object, Process instance

        y : array, shape (x, y, (z))
            True labels for X.

        sample_weight : array, shape(x, y, (z)). Optional, default: None
            Sample weights.

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

        mdl = _sklearn_gaussiannb(priors=self.priors)

        y_2d = np.reshape(y, functools.reduce(operator.mul, y.shape, 1))

        mdl.fit(self._X.flatten(), y_2d, sample_weight=sample_weight)

        self.mdl = mdl
        self.class_prior_ = mdl.class_prior_
        self.class_count_ = mdl.class_count_
        self.theta_ = mdl.theta_
        self.sigma_ = mdl.sigma_

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

        y_pred = self.mdl.predict(X.flatten())
        y_pred = np.reshape(y_pred, X.shape[:-1])

        return y_pred

    def predict_log_proba(self, X):
        """Compute log probabilities of possible outcomes for samples in X.

        The model need to have probability information computed at training
        time: fit with attribute probability set to True.

        Parameters
        ----------
        X : object, Process instance

        Returns
        -------
        T : array, shape (x, y, (z), n_classes)
            Returns the log-probabilities of the sample for each class in the model.
            The columns correspond to the classes in sorted order, as they appear in
            the attribute classes_.

        """
        if not isinstance(X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        T = self.mdl.predict_log_proba(X.flatten())
        T = np.reshape(T, X.shape[:-1] + (T.shape[1], ))

        return T

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
