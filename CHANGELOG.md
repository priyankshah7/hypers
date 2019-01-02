# hypers changelog

## Version 0.0.11 (pre-release)
#### Features 
+ Can now multiply, divide, add and subtract image array from data 
(e.g. (300,300,1000) * (300,300)) directly on Dataset
+ Printing the Dataset instance now prints information about the stored data

## Version 0.0.10 (pre-release)
#### Features
+ Added MeanShift, AffinityPropagation and DBSCAN to the cluster method in Dataset
+ Added Normalizer to preprocess method in Dataset
+ Added following magic methods to Dataset:
    + __setitem__
    + __mul__
    + __truediv__
    + __add__
    + __sub__
 + Added plotting for cluster, decompose and scree
 + Added vertex component analysis as vca method in Dataset
 + Added unconstrained least-squares spectra fitting in abundance method in Dataset
 + Added Gaussian mixture models in the mixture method in Dataset
  
#### Performance enhancements
+ Looping over dims using ndindex, ndenumerate (speed improvement)

## Version 0.0.9 (pre-release)
#### Features
+ Renamed package to hypers

## Version 0.0.8 (pre-release)
#### Features
+ Added preprocess method to Process class which takes preprocessing classes from sklearn to preprocess data.

## Version 0.0.7 (pre-release)
#### Features
+ Removed skhyper wrappers for scikit-learn classes. Added new decompose and cluster methods to the Process class to which a scikit-learn class is passed.
+ Removed skhyper wrappers for classification. This will be added again in a similar vein to the above decompose and cluster methods in the future.

## Version 0.0.6 (pre-release)
#### Features
+ Added a plotting package for cluster/decomposition techniques and Process object

#### Performance enhancements
+ Moved normalization and smoothing directly to the Process class.

#### Bug fixes
+ Corrected issue with updating spectrum in the hyperspectral viewer for asymmetrical images.

## Version 0.0.5 (pre-release)
#### Features
+ Added MLPClassifier in `skhyper.neural_network`

#### Miscellaneous
+ Added tests for MLPClassifier

## Version 0.0.4 (pre-release)
#### Features

#### Performance enhancements
+ Decomposition techniques return a list of arrays for spec_components and image_components

#### Bug fixes
+ Corrected issue with output of predict() in reshaping array (issue with the classifiers)

#### Miscellaneous
+ Added tests for SVC, GaussianNB and KNeighborsClassifier
+ Changed test dataset from random to sklearn's make_blob


## Version 0.0.3 (pre-release)
#### Features
+ Added GaussianNB in `skhyper.naive_bayes`
+ Added SVC in `skhyper.svm`
+ Added KNeighborsClassifier in `skhyper.neighbors`
+ Scree plot array can be accessed directly from Process object