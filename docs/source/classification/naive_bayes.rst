===========
Naive Bayes
===========

Naive Bayes methods are a set of supervised learning algorithms based
on applying Bayes’ theorem with the “naive” assumption of independence
between every pair of features.

As of now, the implementation only supports classification using the
Gaussian Naive Bayes algorithm, :class:`~skhyper.naive_bayes.GaussianNB`.

Naive Bayes learners and classifiers can be extremely fast compared to
more sophisticated methods. The decoupling of the class conditional
feature distributions means that each distribution can be independently
estimated as a one dimensional distribution. This in turn helps to
alleviate problems stemming from the curse of dimensionality.

On the flip side, although naive Bayes is known as a decent classifier,
it is known to be a bad estimator, so the probability outputs from
``predict_proba`` are not to be taken too seriously.

**Example**

.. code-block:: python

    import numpy as np
    from skhyper.process import Process
    from skhyper.naive_bayes import GaussianNB

    # Note that this is a demonstration of syntax. This example won't provide
    # any sensible results due to the randomized data

    random_dataset = np.random.rand(250, 250, 1024)
    random_labels = np.random.randit(4, size=(250, 250))
    random_test = np.random.rand(250, 250, 1024)

    # random_labels contains 4 class labels

    X = Process(random_data, scale=True)
    X_test = Process(random_test, scale=True)

    mdl = GaussianNB()
    mdl.fit(X, random_labels)

    # To use the trained model to predict test dataset:
    mdl.predict(X_test)
