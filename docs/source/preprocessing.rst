=============
Preprocessing
=============

Preprocessing data with ``scikit-hyper`` is done directly on the Process object model. At 
the moment, the following preprocessing classes from ``scikit-learn`` are supported:

- MaxAbsScaler
- MinMaxScaler
- PowerTransformer
- QuantileTransformer
- RobustScaler
- StandardScaler

.. code-block:: python

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from skhyper.process import Process

    tst_data = np.random.rand(50, 50, 100)
    X = Process(tst_data)

    X.preprocess(
        mdl=StandardScaler()
    )
