# scikit-hyper changelog

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