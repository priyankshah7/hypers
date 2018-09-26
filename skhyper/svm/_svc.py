"""
C-support vector classification
"""
import operator
import functools
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC as _sklearn_svc
from sklearn.model_selection import train_test_split

from skhyper.process import Process


class SVC:
    """C-Support Vector Classification

    The implementation is based on libsvm. The fit time complexity
    is more than quadratic with the number of samples which makes it hard
    to scale to dataset with more than a couple of 10000 samples.

    The multiclass support is handled according to a one-vs-one scheme.

    Parameters
    ----------
    C : float, optional (default=1.0)
        Penalty parameter C of the error term.

    kernel : string, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or
        a callable.
        If none is given, 'rbf' will be used. If a callable is given it is
        used to pre-compute the kernel matrix from data matrices; that matrix
        should be an array of shape ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default='auto')
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If gamma is 'auto' then 1/n_features will be used instead.

    coef0 : float, optional (default=0.0)
        Independent term in kernel function.
        It is only significant in 'poly' and 'sigmoid'.

    probability : boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.

    shrinking : boolean, optional (default=True)
        Whether to use the shrinking heuristic.

    tol : float, optional (default=1e-3)
        Tolerance for stopping criterion.

    cache_size : float, optional
        Specify the size of the kernel cache (in MB).

    class_weight : {dict, 'balanced'}, optional
        Set the parameter C of class i to class_weight[i]*C for
        SVC. If not given, all classes are supposed to have
        weight one.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

    verbose : bool, default: False
        Enable verbose output. Note that this setting takes advantage of a
        per-process runtime setting in libsvm that, if enabled, may not work
        properly in a multithreaded context.

    max_iter : int, optional (default=-1)
        Hard limit on iterations within solver, or -1 for no limit.

    decision_function_shape : 'ovo', 'ovr', default='ovr'
        Whether to return a one-vs-rest ('ovr') decision function of shape
        (n_samples, n_classes) as all other classifiers, or the original
        one-vs-one ('ovo') decision function of libsvm which has shape
        (n_samples, n_classes * (n_classes - 1) / 2).

    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    support_ : array-like, shape = [n_SV]
        Indices of support vectors.

    support_vectors_ : array-like, shape = [n_SV, n_features]
        Support vectors.

    n_support_ : array-like, dtype=int32, shape = [n_class]
        Number of support vectors for each class.

    dual_coef_ : array, shape = [n_class-1, n_SV]
        Coefficients of the support vector in the decision function.
        For multiclass, coefficient for all 1-vs-1 classifiers.
        The layout of the coefficients in the multiclass case is somewhat
        non-trivial. See the section about multi-class classification in the
        SVM section of the User Guide for details.

    coef_ : array, shape = [n_class-1, n_features]
        Weights assigned to the features (coefficients in the primal
        problem). This is only available in the case of a linear kernel.

        `coef_` is a readonly property derived from `dual_coef_` and
        `support_vectors_`.

    intercept_ : array, shape = [n_class * (n_class-1) / 2]
        Constants in decision function.

    """
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True,
                 probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
                 max_iter=-1, decision_function_shape='ovr', random_state=None):
        self._X = None

        # sklearn SVC model
        self.mdl = None

        # sklearn SVC attributes
        self.support_ = None
        self.support_vectors_ = None
        self.n_support_ = None
        self.dual_coef_ = None
        self.coef_ = None
        self.intercept_ = None

        # sklearn optional SVC arguments
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

    def _check_is_fitted(self):
        """
        Performs a check to see if self.data is empty. If it is, then fit() has not been called yet.
        """
        if self.mdl is None:
            raise AttributeError('Data has not yet been fitted with fit()')

    def fit_train_test(self, X, y, test_size=0.8):
        if not isinstance(X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        if type(y) != np.ndarray:
            raise TypeError('Target value array must be a numpy array')

        if len(y.shape) != 2 and len(y.shape) != 3:
            raise TypeError('Target value array must be 2- or 3-dimensional.')

        X_train, X_test, y_train, y_test = train_test_split(X.flatten(),
                                                            np.reshape(y, functools.reduce(operator.mul, y.shape, 1)),
                                                            test_size=test_size)

        le = preprocessing.LabelEncoder()
        le.fit(y_train)
        y_train = le.transform(y_train)

        le.fit(y_test)
        y_test = le.transform(y_test)

        mdl = _sklearn_svc(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                           coef0=self.coef0, shrinking=self.shrinking, probability=self.probability,
                           tol=self.tol, cache_size=self.cache_size, class_weight=self.class_weight,
                           verbose=self.verbose, max_iter=self.max_iter,
                           decision_function_shape=self.decision_function_shape, random_state=self.random_state)

        mdl.fit(X_train, y_train)

        self.mdl = mdl
        score = mdl.score(X_test, y_test)

        print('Score of the trained model on the test data is: ' + str(score))
        return score

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

        mdl = _sklearn_svc(C=self.C, kernel=self.kernel, degree=self.degree, gamma=self.gamma,
                           coef0=self.coef0, shrinking=self.shrinking, probability=self.probability,
                           tol=self.tol, cache_size=self.cache_size, class_weight=self.class_weight,
                           verbose=self.verbose, max_iter=self.max_iter,
                           decision_function_shape=self.decision_function_shape, random_state=self.random_state)

        y_2d = np.reshape(y, functools.reduce(operator.mul, y.shape, 1))

        mdl.fit(self._X.flatten(), y_2d, sample_weight=sample_weight)

        self.mdl = mdl
        self.support_ = mdl.support_
        self.support_vectors_ = mdl.support_vectors_
        self.n_support_ = mdl.n_support_
        self.dual_coef_ = mdl.dual_coef_
        if self.kernel == 'linear': self.coef_ = mdl.coef_
        self.intercept_ = mdl.intercept_

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

        self._check_is_fitted()

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
