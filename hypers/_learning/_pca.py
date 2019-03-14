import hypers as hp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class _pca:
    def __init__(self, X: 'hp.Dataset'):
        self.X = X

    def scree(self):
        mdl = PCA()
        mdl.fit_transform(self.X.flatten())

        return mdl.explained_variance_ratio_

    def plot_scree(self):
        mdl = PCA()
        mdl.fit_transform(self.X.flatten())

        plt.plot(mdl.explained_variance_ratio_)
        plt.xlabel('Principal components')
        plt.ylabel('Variance ratio')
        plt.title('Scree plot')
        plt.tight_layout()
        plt.show()

    def calculate(self, n_components=5):
        mdl = PCA(n_components=n_components)
        ims = mdl.fit_transform(self.X.flatten()).reshape(self.X.data.shape[:-1] + (n_components,))
        spcs = mdl.components_.transpose()

        return ims, spcs

