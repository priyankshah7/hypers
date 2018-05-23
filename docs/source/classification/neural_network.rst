=======================================
Neural network (multi-layer perceptron)
=======================================

:class:`~skhyper.neural_network.MLPClassifier` is a supervised learning algorithm that learns
a function by training on a dataset. Given a set of features
:math:`X = x_{1}, x_{2}, ..., x_{m}` and a target :math:`y`, it can learn
a non-linear function approximator for classification (for the case of
this module).

.. warning::

    This implementation is not intended for large-scale applications.
    In particular, scikit-learn (which this module wraps around) offers no GPU support.

**Example**

.. code-block:: python

    import numpy as np
    from skhyper.process import Process
    from skhyper.neural_network import MLPClassifier

    # Note that this is a demonstration of syntax. This example won't provide
    # any sensible results due to the randomized data

    random_dataset = np.random.rand(250, 250, 1024)
    random_labels = np.random.randit(4, size=(250, 250))
    random_test = np.random.rand(250, 250, 1024)

    # random_labels contains 4 class labels

    X = Process(random_data, scale=True)
    X_test = Process(random_test, scale=True)

    mdl = MLPClassifier()
    mdl.fit(X, random_labels)

    # To use the trained model to predict test dataset:
    mdl.predict(X_test)
