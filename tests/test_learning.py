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
