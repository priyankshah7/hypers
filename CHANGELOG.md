# scikit-hyper changelog

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