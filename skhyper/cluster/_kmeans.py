"""
K-means clustering
"""
import numpy as np
from sklearn.cluster import KMeans as _sklearn_kmeans

from skhyper.process import Process


class KMeans:
    """
    K-means clustering

    Parameters
    ----------
    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence

    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances

        False : never precompute distances

    verbose : int, default 0
        Verbosity mode.

    random_state : int, RandomState instance or None, optional, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    copy_x : boolean, default True
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True, then the original data is not
        modified.  If False, the original data is modified, and put back before
        the function returns, but small numerical differences may be introduced
        by subtracting and then adding the data mean.

    n_jobs : int
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    algorithm : "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.

    Attributes
    ----------
    image_components_ : list, size (n_clusters)
        Each element of the list contains the corresponding image for the cluster.

    spec_components_ : list, size (n_clusters)
        Each element of the list contains the corresponding spectrum for the cluster.

    mdl : KMeans() instance
        An instance of the KMeans model from scikit-learn used when instantiating KMeans() from scikit-hyper

    labels_ : array
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.


    Examples
    --------
    >>> import numpy as np
    >>> from skhyper.cluster import KMeans
    >>> from skhyper.process import Process
    >>>
    >>> data = np.random.rand(100, 100, 10, 1024)
    >>> X = Process(data)
    >>>
    >>> mdl = KMeans(n_clusters=4)
    >>> mdl.fit(X)
    >>>
    >>> # Access image and spectral components of the individual clusters
    >>> mdl.image_components_[2]
    >>> mdl.spec_components_[2]
    """
    def __init__(self, n_clusters, init='k-means++', n_init=10, max_iter=300,
                 tol=1e-4, precompute_distances='auto', verbose=0, random_state=None,
                 copy_x=True, n_jobs=-1, algorithm='auto'):
        self._X = None

        # sklearn KMeans model
        self.mdl = None

        # clusters
        self.image_components_ = None
        self.spec_components_ = None

        # sklearn KMeans attributes
        self.labels_ = None
        self.inertia_ = None

        # sklearn optional KMeans arguments
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def fit(self, X):
        """
        Fits the KMeans model to the processed hyperspectral array.

        Parameters
        ----------
        X : object, type (Process)

        Returns
        -------
        self : object
            Returns self.

        """
        self._X = X
        if not isinstance(self._X, Process):
            raise TypeError('Data needs to be passed to skhyper.process.Process first')

        mdl = _sklearn_kmeans(n_clusters=self.n_clusters, init=self.init, n_init=self.n_init,
                              max_iter=self.max_iter, tol=self.tol,
                              precompute_distances=self.precompute_distances,
                              verbose=self.verbose, random_state=self.random_state, copy_x=self.copy_x,
                              n_jobs=self.n_jobs, algorithm=self.algorithm).fit(self._X.flatten())

        labels = np.reshape(mdl.labels_, self._X.shape[:-1])
        labels += 1

        self.mdl = mdl
        self.labels_ = labels
        self.inertia_ = mdl.inertia_

        self.image_components_, self.spec_components_ = [0]*self.n_clusters, [0]*self.n_clusters
        for cluster in range(self.n_clusters):
            self.image_components_[cluster] = np.squeeze(np.where(labels == cluster + 1, labels, 0) / (cluster + 1))

            self.spec_components_[cluster] = np.zeros(self._X.shape)
            for spectral_point in range(self._X.shape[-1]):
                self.spec_components_[cluster][..., spectral_point] = np.multiply(self._X[..., spectral_point],
                                                                                  np.where(labels == cluster + 1, labels, 0) / (cluster + 1))

            if self._X.ndim == 3:
                self.spec_components_[cluster] = np.squeeze(np.mean(np.mean(self.spec_components_[cluster], 1), 0))

            elif self._X.ndim == 4:
                self.spec_components_[cluster] = np.squeeze(np.mean(np.mean(np.mean(self.spec_components_[cluster], 2), 1), 0))

        return self
