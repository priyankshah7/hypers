=========================
Hyperspectral data viewer
=========================

Included with `scikit-hyper` is a hyperspectral data viewer that allows for
visualization and interactivity with the hyperspectral dataset. It can be
accessed in 2 ways:

- Using :class:`~skhyper.view.hsiPlot` directly.

    .. code-block:: python

        import numpy as np
        from skhyper.process import Process
        from skhyper.view import hsiPlot

        test_data = np.random.rand(100, 100, 5, 512)
        X = Process(test_data)

        hsiPlot(X)


- From the :class:`~skhyper.process.Process` instance variable:

    .. code-block:: python

        import numpy as np
        from skhyper.process import Process

        test_data = np.random.rand(100, 100, 5, 512)
        X = Process(test_data)

        X.view()


The hyperspectral data viewer is a lightweight pyqt gui. Below is an example:

.. figure:: hyperspectral_view.png
    :width: 600px
    :align: center
    :alt: Hyperspectral viewer
    :figclass: align-center

    Hyperspectral data viewer.

.. note::

    If using `scikit-hyper` in a Jupyter notebook, it is still possible to use
    the data viewer. However the notebook cell will be frozen until the data
    viewer has been closed.

    This is due to the fact that the data viewer uses the same CPU process as the
    notebook. This may be changed in the future.
