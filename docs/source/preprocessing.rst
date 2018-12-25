=============
Preprocessing
=============

Preprocessing data with ``hypers`` is performed directly on the Dataset object. At
the moment, the following preprocessing classes from ``scikit-learn`` are supported:

- MaxAbsScaler
- MinMaxScaler
- PowerTransformer
- QuantileTransformer
- RobustScaler
- StandardScaler

.. code-block:: python

    import numpy as np
    import hypers as hp
    from sklearn.preprocessing import StandardScaler

    tst_data = np.random.rand(50, 50, 100)
    X = hp.Dataset(tst_data)

    X.preprocess(
        mdl=StandardScaler()
    )
