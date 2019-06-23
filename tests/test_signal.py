import numpy as np
import hypers as hp
from hypers.signal.anscombe import anscombe_transformation, inverse_anscombe_transformation


class TestSignal:
    def setup(self):
        self.n1 = np.random.rand(30)
        self.h1 = hp.hparray(self.n1)

    def test_anscombe(self):
        nasignal = anscombe_transformation(self.n1, 0.5, 0.5, 1)
        hasignal = anscombe_transformation(self.h1, 0.5, 0.5, 1)

        assert nasignal.shape == self.n1.shape
        assert hasignal.shape == self.h1.shape

        nisignal = inverse_anscombe_transformation(nasignal, 0.5, 0.5, 1)
        hisignal = inverse_anscombe_transformation(hasignal, 0.5, 0.5, 1)

        assert nisignal.shape == self.n1.shape
        assert hisignal.shape == self.h1.shape
