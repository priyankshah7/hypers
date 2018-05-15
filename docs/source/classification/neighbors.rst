====================
K-nearest neighbors
====================

Neighbors-based classification is a type of `instance-based learning`
or `non-generalizing learning`: it does not attempt to construct a
general internal model, but simply stores instances of the training data.
Classification is computed from a simple majority vote of the nearest
neighbors of each point: a query point is assigned the data class which
has the most representatives within the nearest neighbors of the point.

:class:`~skhyper.neighbors.KNeighborsClassifier` implements learning based
on the ``k`` nearest neighbors of each query point, where ``k`` is an integer
value specified by the user. The optimal value of ``k`` is highly
data-dependent: in general a larger ``k`` supresses the effects of nouse,
but makes the classification boundaries less distinct.

**Example**

.. code-block:: python

    import numpy as np
    from skhyper.process import Process
    from skhyper.neighbors import KNeighborsClassifier

    # Note that this is a demonstration of syntax. This example won't provide
    # any sensible results due to the randomized data

    random_dataset = np.random.rand(250, 250, 1024)
    random_labels = np.random.randit(4, size=(250, 250))
    random_test = np.random.rand(250, 250, 1024)

    # random_labels contains 4 class labels

    X = Process(random_data, scale=True)
    X_test = Process(random_test, scale=True)

    mdl = KNeighborsClassifier()
    mdl.fit(X, random_labels)

    # To use the trained model to predict test dataset:
    mdl.predict(X_test)
