The `hyperanalysis.cluster` module provides 3 clustering techniques for hyperspectral data. All 3 are
implemented using the `scikit-learn` module and follow a similar syntax to the module.

## K-means
Implements k-means clustering on hyperspectral data using the `scikit-learn` module. The same parameters
available in `scikit-learn` can be used here:
```python
class hyperanalysis.cluster. KMeans(n_clusters, init='k-means++', n_init=10, max_iter=300,
                 tol=1e-4, precompute_distances='auto', verbose=0, random_state=None,
                 copy_x=True, n_jobs=-1, algorithm='auto')
```
**Parameters**

+ `n_clusters` : </br>
    int, must be specified.

    Number of clusters required. </br></br>

+ `init` : </br>
    {‘k-means++’, ‘random’ or an ndarray}

    Method for initialization, defaults to ‘k-means++’:

    ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.

    ‘random’: choose k observations (rows) at random from data for the initial centroids.

    If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers. </br></br>

+ `n_inits` : </br>
    int, default: 10

    Number of time the k-means algorithm will be run with different centroid seeds.
    The final results will be the best output of n_init consecutive runs in terms of inertia. </br></br>

+ `max_iter` : </br>
    int, default: 300

    Maximum number of iterations of the k-means algorithm for a single run. </br></br>

+ `tol` : </br>
    float, default: 1e-4

    Relative tolerance with regards to inertia to declare convergence </br></br>

+ `precompute_distances` : </br>
    {‘auto’, True, False}

    Precompute distances (faster but takes more memory).

    ‘auto’ : do not precompute distances if n_samples * n_clusters > 12 million. This corresponds to about 100MB overhead per job using double precision.

    True : always precompute distances

    False : never precompute distances </br></br>

+ `verbose` : </br>
    int, default 0

    Verbosity mode. </br></br>

+ `random_state` : </br>
    int, RandomState instance or None, optional, default: None

    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used by np.random. </br></br>

+ `copy_x` : </br>
    boolean, default True

    When pre-computing distances it is more numerically accurate to center the data first.
    If copy_x is True, then the original data is not modified.
    If False, the original data is modified, and put back before the function returns,
    but small numerical differences may be introduced by subtracting and then adding the data mean. </br></br>

+ `n_jobs` : </br>
    int

    The number of jobs to use for the computation. This works by computing each of the n_init
    runs in parallel.

    If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all,
    which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used. </br></br>

+ `algorithm` : </br>
    “auto”, “full” or “elkan”, default=”auto”

    K-means algorithm to use. The classical EM-style algorithm is “full”. The “elkan” variation
    is more efficient by using the triangle inequality, but currently doesn’t support sparse data.
    “auto” chooses “elkan” for dense data and “full” for sparse data. </br>


**Attributes**

+ `labels` : </br>
    array

    Returns a 2D/3D image with each pixel labeled according to the assigned cluster. `fit()` must
    be called first prior to accessing.</br></br>

+ `data_clusters` : </br>
    list (n_clusters)

    Each element of the list contains an array containing the original data, but cropped to include
    only a specific cluster. `fit()` must be called first prior to accessing.


**Methods**

+ `fit(X)` : </br>
    `X` : array, either in form `[x, y, spectrum]` or `[x, y, z, spectrum]`.

    Fits the k-means algorithm to the hyperspectral data array. </br></br>

+ `plot()` : </br>
    Outputs a `matplotlib` plot showing the image and spectrum associated to each cluster. `fit()`
    must be called first prior to accessing.


**Example**
```python
import numpy as np
from hyperanalysis.cluster import KMeans

test_data = np.random.rand(100, 100, 1024)
mdl = KMeans(n_clusters=3)
mdl.fit(test_data)
```
</br>

## Agglomerative clustering
Implements agglomerative clustering on hyperspectral data using the `scikit-learn` module.
The same parameters available in `scikit-learn` can be used here:
```python
class hyperanalysis.cluster. AgglomerativeClustering(n_clusters, affinity='euclidean',
                    memory=None, connectivity=None, compute_full_tree='auto', linkage='ward',
                    pooling_func=np.mean)
```
**Parameters**

+ `n_clusters` : </br>
    int, must be specified.

    Number of clusters required. </br></br>

+ `affinity` : </br>
    string or callable, default: “euclidean”

    Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”,
    or ‘precomputed’. If linkage is “ward”, only “euclidean” is accepted. </br></br>

+ `memory` : </br>
    None, str or object with the joblib.Memory interface, optional

    Used to cache the output of the computation of the tree. By default, no caching is done.
    If a string is given, it is the path to the caching directory. </br></br>

+ `connectivity` : </br>
    array-like or callable, optional

    Connectivity matrix. Defines for each sample the neighboring samples following a given
    structure of the data. This can be a connectivity matrix itself or a callable that transforms
    the data into a connectivity matrix, such as derived from kneighbors_graph.
    Default is None, i.e, the hierarchical clustering algorithm is unstructured. </br></br>

+ `compute_full_tree` : </br>
    bool or ‘auto’ (optional)

    Stop early the construction of the tree at n_clusters. This is useful to decrease
    computation time if the number of clusters is not small compared to the number of samples.
    This option is useful only when specifying a connectivity matrix. Note also that when varying
    the number of clusters and using caching, it may be advantageous to compute the full tree. </br></br>

+ `linkage` : </br>
    {“ward”, “complete”, “average”}, optional, default: “ward”

    Which linkage criterion to use. The linkage criterion determines which distance to use between
    sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion.

     + ward minimizes the variance of the clusters being merged.
     + average uses the average of the distances of each observation of the two sets.
     + complete or maximum linkage uses the maximum distances between all observations of the two sets. </br></br>

+ `pooling_func` : </br>
    callable, default=np.mean

    This combines the values of agglomerated features into a single value, and should accept
    an array of shape [M, N] and the keyword argument axis=1, and reduce it to an array of size [M]. </br></br>


**Attributes**

+ `labels` : </br>
    array

    Returns a 2D/3D image with each pixel labeled according to the assigned cluster. </br></br>

+ `data_clusters` : </br>
    list (n_clusters)

    Each element of the list contains an array containing the original data, but cropped to include
    only a specific cluster.


**Methods**

+ `fit(X)` : </br>
    `X` : array, either in form `[x, y, spectrum]` or `[x, y, z, spectrum]`.

    Fits the agglomerative clustering algorithm to the hyperspectral data array. </br></br>

+ `plot()` : </br>
    Outputs a `matplotlib` plot showing the image and spectrum associated to each cluster. `fit()`
    must be called first prior to accessing.


**Example**
```python
import numpy as np
from hyperanalysis.cluster import AgglomerativeClustering

test_data = np.random.rand(100, 100, 1024)
mdl = AgglomerativeClustering(n_clusters=3)
mdl.fit(test_data)
```
</br>

## Spectral clustering
Implements spectral clustering on hyperspectral data using the `scikit-learn` module.
The same parameters available in `scikit-learn` can be used here:
```python
class hyperanalysis.cluster. SpectralClustering(n_clusters, eigen_solver=None, random_state=None,
                    n_init=10, gamma=1., affinity='rbf', n_neighbors=10, eigen_tol=0.0,
                    assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1)
```
**Parameters**

+ `n_clusters` : </br>
    int, must be specified.

    Number of clusters required. </br></br>

+ `eigen_solver` : </br>
    {None, ‘arpack’, ‘lobpcg’, or ‘amg’}

    The eigenvalue decomposition strategy to use. AMG requires pyamg to be installed.
    It can be faster on very large, sparse problems, but may also lead to instabilities. </br></br>

+ `random_state` : </br>
    int, RandomState instance or None, optional, default: None

    A pseudo random number generator used for the initialization of the lobpcg eigen vectors
    decomposition when eigen_solver == ‘amg’ and by the K-Means initialization.
    If int, random_state is the seed used by the random number generator; If RandomState instance,
    random_state is the random number generator; If None, the random number generator is the
    RandomState instance used by np.random. </br></br>

+ `n_init` : </br>
    int, optional, default: 10

    Number of time the k-means algorithm will be run with different centroid seeds.
    The final results will be the best output of n_init consecutive runs in terms of inertia. </br></br>

+ `gamma` : </br>
    float, default=1.0

    Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels.
    Ignored for `affinity='nearest_neighbors'`. </br></br>

+ `affinity` : </br>
    string, array-like or callable, default ‘rbf’

    If a string, this may be one of ‘nearest_neighbors’, ‘precomputed’, ‘rbf’ or one of the
    kernels supported by sklearn.metrics.pairwise_kernels.

    Only kernels that produce similarity scores (non-negative values that increase with similarity)
    should be used. This property is not checked by the clustering algorithm. </br></br>

+ `n_neighbors` : </br>
    integer

    Number of neighbors to use when constructing the affinity matrix using the nearest
    neighbors method. Ignored for `affinity='rbf'`. </br></br>

+ `eigen_tol` : </br>
    float, optional, default: 0.0

    Stopping criterion for eigendecomposition of the Laplacian matrix when using arpack eigen_solver. </br></br>

+ `assign_labels` : </br>
    {‘kmeans’, ‘discretize’}, default: ‘kmeans’

    The strategy to use to assign labels in the embedding space. There are two ways to
    assign labels after the laplacian embedding. k-means can be applied and is a popular choice.
    But it can also be sensitive to initialization. Discretization is another approach which is
    less sensitive to random initialization. </br></br>

+ `degree` : </br>
    float, default=3

    Degree of the polynomial kernel. Ignored by other kernels. </br></br>

+ `coef0` : </br>
    float, default=1

    Zero coefficient for polynomial and sigmoid kernels. Ignored by other kernels. </br></br>

+ `kernel_params` : </br>
    dictionary of string to any, optional

    Parameters (keyword arguments) and values for kernel passed as callable object.
    Ignored by other kernels. </br></br>

+ `n_jobs` : </br>
    int, optional (default = 1)

    The number of parallel jobs to run. If `-1`, then the number of jobs is set to the number of
    CPU cores. </br></br>


**Attributes**

+ `labels` : </br>
    array

    Returns a 2D/3D image with each pixel labeled according to the assigned cluster. </br></br>

+ `data_clusters` : </br>
    list (n_clusters)

    Each element of the list contains an array containing the original data, but cropped to include
    only a specific cluster.


**Methods**

+ `fit(X)` : </br>
    `X` : array, either in form `[x, y, spectrum]` or `[x, y, z, spectrum]`.

    Fits the spectral clustering algorithm to the hyperspectral data array. </br></br>

+ `plot()` : </br>
    Outputs a `matplotlib` plot showing the image and spectrum associated to each cluster. `fit()`
    must be called first prior to accessing.


**Example**
```python
import numpy as np
from hyperanalysis.cluster import SpectralClustering

test_data = np.random.rand(100, 100, 1024)
mdl = SpectralClustering(n_clusters=3)
mdl.fit(test_data)
```
</br>
