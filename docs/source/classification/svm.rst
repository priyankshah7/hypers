=======================
Support vector machines
=======================

The implementation of support vector machines (SVM) is `scikit-hyper` only supports
classification (i.e. no regression). It is provided in the class :class:`skhyper.svm.SVC`.

The advantages of support vector machines for hyperspectral data include:

- Effective in high dimensional spaces
- Still effective in cases where number of dimensions (spectral features) is
greater than the number of samples
- Uses a subset of training points in the decision function, so it is also memory efficient

The primary disadvantage though, is that it does not directly provide probability estimates.

**Example**

.. code-block:: python

    import numpy as np
    from skhyper.process import Process
    from skhyper.svm import SVC

    # Note that this is a demonstration of syntax. This example won't provide
    # any sensible results due to the randomized data

    random_dataset = np.random.rand(250, 250, 1024)
    random_labels = np.random.randit(4, size=(250, 250))
    random_test = np.random.rand(250, 250, 1024)

    # random_labels contains 4 class labels

    X = Process(random_data, scale=True)
    X_test = Process(random_test, scale=True)

    mdl = SVC()
    mdl.fit(X, random_labels)

    # To use the trained model to predict test dataset:
    mdl.predict(X_test)
