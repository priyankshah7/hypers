import numpy as np
import hypers as hp


class TestLearning:
    def setup(self):
        self.n3 = np.random.rand(10, 10, 30)
        self.n4 = np.random.rand(10, 10, 10, 30)
        self.n5 = np.random.rand(10, 10, 10, 2, 30)

        self.h3 = hp.hparray(self.n3)
        self.h4 = hp.hparray(self.n4)
        self.h5 = hp.hparray(self.n5)

        self.arrays = (self.h3, self.h4, self.h5)

    def test_decompose(self):
        for array in self.arrays:
            # Testing pca, ica, and nmf methods
            pca = array.decompose.pca
            ica = array.decompose.ica
            nmf = array.decompose.nmf
            for dmethod in (pca, ica, nmf):
                assert dmethod.ims is None
                assert dmethod.spcs is None

                _, _ = dmethod.calculate(n_components=None)
                assert dmethod.ims.shape == array.shape
                assert dmethod.spcs.shape == (array.shape[-1], array.shape[-1])

                _, _ = dmethod.calculate(n_components=4)
                assert dmethod.ims.shape == array.shape[:-1] + (4,)
                assert dmethod.spcs.shape == (array.shape[-1], 4)

            assert array.decompose.pca.scree().shape[0] == array.shape[-1]

            # Testing vca method
            assert array.decompose.vca.spcs is None
            assert array.decompose.vca.coords is None

            _, _ = array.decompose.vca.calculate(n_components=3)
            assert array.decompose.vca.spcs.shape == (array.shape[-1], 3)
            assert isinstance(array.decompose.vca.coords, list)
            assert len(array.decompose.vca.coords) == 3

    def test_cluster(self):
        for array in self.arrays:
            _, _ = array.cluster.kmeans.calculate(n_clusters=3, decomposed=False)
            assert array.cluster.kmeans.labels.shape == array.shape[:-1]
            assert np.unique(array.cluster.kmeans.labels).shape[0] == 3
            assert array.cluster.kmeans.spcs.shape == (array.shape[-1], 3)

            _, _ = array.cluster.kmeans.calculate(n_clusters=3, decomposed=True, pca_comps=4)
            assert array.cluster.kmeans.labels.shape == array.shape[:-1]
            assert np.unique(array.cluster.kmeans.labels).shape[0] == 3
            assert array.cluster.kmeans.spcs.shape == (array.shape[-1], 3)

    def test_abundance(self):
        for array in self.arrays:
            ucls = array.abundance.ucls
            nnls = array.abundance.nnls
            fcls = array.abundance.fcls

            for amethod in (ucls, nnls, fcls):
                spec1d = np.random.rand(array.shape[-1])
                _ = amethod.calculate(spec1d)
                assert amethod.map.shape == array.shape[:-1] + (1,)

                spec2d = np.random.rand(array.shape[-1], 3)
                _ = amethod.calculate(spec2d)
                assert amethod.map.shape == array.shape[:-1] + (3,)

    def test_mixture(self):
        for array in self.arrays:
            gaussian = array.mixture.gaussian_mixture

            for mmethod in (gaussian,):
                _, _ = mmethod.calculate(n_components=None)
                assert mmethod.labels.shape == array.shape[:-1]
                assert mmethod.spcs.shape == (array.shape[-1], array.shape[-1])

                _, _ = mmethod.calculate(n_components=3)
                assert mmethod.labels.shape == array.shape[:-1]
                assert mmethod.spcs.shape == (array.shape[-1], 3)
