from typing import Union
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler,
    StandardScaler, Normalizer
)
from sklearn.cluster import (
    KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
)
from sklearn.decomposition import (
    PCA, FastICA, IncrementalPCA, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning,
    FactorAnalysis, NMF, LatentDirichletAllocation
)

PREPROCESSING_TYPES = (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    Normalizer
)

CLUSTER_TYPES = (
    KMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN
)

DECOMPOSE_TYPES = (
    PCA,
    FastICA,
    #KernelPCA,
    IncrementalPCA,
    TruncatedSVD,
    DictionaryLearning,
    MiniBatchDictionaryLearning,
    FactorAnalysis,
    NMF,
    LatentDirichletAllocation
)


PreprocessType = Union[
    MaxAbsScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler, Normalizer
]

ClusterType = Union[
    KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
]

DecomposeType = Union[
    PCA, FastICA, IncrementalPCA, TruncatedSVD, DictionaryLearning, MiniBatchDictionaryLearning, NMF,
    FactorAnalysis, LatentDirichletAllocation
]
