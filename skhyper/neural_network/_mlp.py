"""
Multi-layer perceptron classification
"""
import operator
import functools
import numpy as np
from sklearn.neural_network import MLPClassifier as _sklearn_mlp

from skhyper.process import Process


class MLPClassifier:
    """Multi-layer perceptron classification

    This model optimizes the log-loss function using LBFGS or stochastic gradient descent.

    Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
        The ith element represents the number of neurons in the ith hidden layer.

    activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
        - 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
        - 'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
        - 'relu', the rectified linear unit function, returns f(x) = max(0, x)

    solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.
        - 'sgd' refers to stochastic gradient descent.
        - 'adam' refers to a stochastic gradient-based optimizer proposed
        by Kingma, Diederik, and Jimmy Ba

        Note: The default solver ‘adam’ works pretty well on relatively
        large datasets (with thousands of training samples or more) in
        terms of both training time and validation score. For small
        datasets, however, ‘lbfgs’ can converge faster and perform better.

    alpha : float, optional, default 0.0001
        L2 penalty (regularization term) parameter.

    batch_size : int, optional, default ‘auto’
        Size of minibatches for stochastic optimizers. If the solver is 'lbfgs',
        the classifier will not use minibatch.
        When set to “auto”, batch_size=min(200, n_samples)

    learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default 'constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by `learning_rate_init`.
        - 'invscaling' gradually decreases the learning rate learning_rate_ at
         each time step ‘t’ using an inverse scaling exponent of `power_t`.
         effective_learning_rate = learning_rate_init / pow(t, power_t)
        - 'adaptive' keeps the learning rate constant to `learning_rate_init`
        as long as training loss keeps decreasing. Each time two consecutive
        epochs fail to decrease training loss by at least tol, or fail to
        increase validation score by at least tol if ‘early_stopping’ is on,
        the current learning rate is divided by 5.

        Only used when solver='sgd'.

    learning_rate_init :double, optional, default 0.001
        The initial learning rate used. It controls the step-size in updating
        the weights. Only used when solver='sgd' or 'adam'.

    power_t : double, optional, default 0.5
        The exponent for inverse scaling learning rate. It is used in updating
        effective learning rate when the `learning_rate` is set to 'invscaling'.
        Only used when solver='sgd'.

    max_iter : int, optional, default 200
        Maximum number of iterations. The solver iterates until
        convergence (determined by `tol`) or this number of iterations.
        For stochastic solvers ('sgd', 'adam'), note that this determines
        the number of epochs (how many times each data point will be used),
        not the number of gradient steps.

    shuffle : bool, optional, default True
        Whether to shuffle samples in each iteration. Only used
        when solver='sgd' or 'adam'.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance
        used by np.random.

    tol : float, optional, default 1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least tol for two consecutive iterations, unless `learning_rate`
        is set to 'adaptive', convergence is considered to be reached and
        training stops.

    verbose : bool, optional, default False
        Whether to print progress messages to stdout.

    warm_start : bool, optional, default False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    momentum : float, default 0.9
        Momentum for gradient descent update. Should be between 0 and 1.
        Only used when solver='sgd'.

    nesterovs_momentum : bool, default True
        Whether to use Nesterov’s momentum. Only used when solver='sgd'
         and momentum > 0.

    early_stopping : bool, default False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training
        when validation score is not improving by at least tol for
        two consecutive epochs. Only effective when solver='sgd' or 'adam'

    validation_fraction : float, optional, default 0.1
        The proportion of training data to set aside as validation set
        for early stopping. Must be between 0 and 1. Only used if
        `early_stopping` is True

    beta_1 : float, optional, default 0.9
        Exponential decay rate for estimates of first moment vector in
        adam, should be in [0, 1]. Only used when solver='adam'

    beta_2 : float, optional, default 0.999
        Exponential decay rate for estimates of second moment vector in
         adam, should be in [0, 1]. Only used when solver='adam'

    epsilon : float, optional, default 1e-8
        Value for numerical stability in adam. Only used when solver='adam'

    Attributes
    ----------
    classes_ : array or list of array of shape (n_classes,)
        Class labels for each output.

    loss_ : float
        The current loss computed with the loss function.

    coefs_ : list, length n_layers - 1
        The ith element in the list represents the weight matrix corresponding
        to layer i.

    intercepts_ : list, length n_layers - 1
        The ith element in the list represents the bias vector corresponding
        to layer i + 1.

    n_iter_ : int,
        The number of iterations the solver has ran.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : str
        Name of the output activation function.

    """
    def __init__(self, hidden_layer_sizes=(100, ), activation='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False,
                 warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        self._X = None

        # sklearn MLPClassifier model
        self.mdl = None

        # sklearn MLPClassifier attributes
        self.classes_ = None
        self.loss_ = None
        self.coefs_ = None
        self.intercepts_ = None
        self.n_iter_ = None
        self.n_layers_ = None
        self.n_outputs_ = None
        self.out_activation_ = None

        # sklearn optional MLPClassifier arguments
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def _check_is_fitted(self):
        """
        Performs a check to see if self.data is empty. If it is, then fit() has not been called yet.
        """
        if self._X is None:
            raise AttributeError('Data has not yet been fitted with fit()')

    def fit(self, X, y):
        """Fit the MLPClassifier model according to the given training data.

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

        mdl = _sklearn_mlp(hidden_layer_sizes=self.hidden_layer_sizes, activation=self.activation,
                           solver=self.solver, alpha=self.alpha, batch_size=self.batch_size,
                           learning_rate=self.learning_rate, learning_rate_init=self.learning_rate_init,
                           power_t=self.power_t, max_iter=self.max_iter, shuffle=self.shuffle,
                           random_state=self.random_state, tol=self.tol, verbose=self.verbose,
                           warm_start=self.warm_start, momentum=self.momentum,
                           nesterovs_momentum=self.nesterovs_momentum, early_stopping=self.early_stopping,
                           validation_fraction=self.validation_fraction, beta_1=self.beta_1,
                           beta_2=self.beta_2, epsilon=self.epsilon)

        y_2d = np.reshape(y, functools.reduce(operator.mul, y.shape, 1))

        mdl.fit(self._X.flatten(), y_2d)

        self.mdl = mdl
        self.classes_ = mdl.classes_
        self.loss_ = mdl.loss_
        self.coefs_ = mdl.coefs_
        self.intercepts_ = mdl.intercepts_
        self.n_iter_ = mdl.n_iter_
        self.n_layers_ = mdl.n_layers_
        self.n_outputs_ = mdl.n_outputs_
        self.out_activation_ = mdl.out_activation_

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
