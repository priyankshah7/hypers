import pytest
import numpy as np
import hypers as hp
from hypers.exceptions import DimensionError

class TestArray:
    def setup(self):
        self.n1 = np.random.rand(30)
        self.n2 = np.random.rand(100, 30)
        self.n3 = np.random.rand(10, 10, 30)
        self.n4 = np.random.rand(10, 10, 10, 30)
        self.n5 = np.random.rand(10, 10, 10, 2, 30)

        self.h1 = hp.array(self.n1)
        self.h2 = hp.array(self.n2)
        self.h3 = hp.array(self.n3)
        self.h4 = hp.array(self.n4)
        self.h5 = hp.array(self.n5)

        self.array_shapes = [
            (30,),
            (100, 30),
            (10, 10, 30),
            (10, 10, 10, 30),
            (10, 10, 10, 2, 30)
        ]

    def test_mean_attributes(self):
        with pytest.raises(AttributeError):
            _ = self.h1.mean_image.shape
        with pytest.raises(AttributeError):
            _ = self.h1.mean_spectrum.shape
        assert self.h2.mean_image.shape == self.array_shapes[1][:-1]
        assert self.h2.mean_spectrum.shape == self.array_shapes[0]
        assert self.h3.mean_image.shape == self.array_shapes[2][:-1]
        assert self.h3.mean_spectrum.shape == self.array_shapes[0]
        assert self.h4.mean_image.shape == self.array_shapes[3][:-1]
        assert self.h4.mean_spectrum.shape == self.array_shapes[0]
        assert self.h5.mean_image.shape == self.array_shapes[4][:-1]
        assert self.h5.mean_spectrum.shape == self.array_shapes[0]

    def test_data_access(self):
        with pytest.raises(AttributeError):
            _ = self.h1.image
        with pytest.raises(AttributeError):
            _ = self.h1.spectrum

        assert self.h2.image[2].shape == self.array_shapes[1][:-1]
        assert self.h2.image[2:10].shape == self.array_shapes[1][:-1]
        assert self.h2.spectrum[2].shape == self.array_shapes[0]
        assert self.h2.spectrum[2:10].shape == self.array_shapes[0]

        assert self.h3.image[2].shape == self.array_shapes[2][:-1]
        assert self.h3.image[2:10].shape == self.array_shapes[2][:-1]
        assert self.h3.spectrum[2, 2].shape == self.array_shapes[0]
        assert self.h3.spectrum[2:10, 2:10].shape == self.array_shapes[0]

        assert self.h4.image[2].shape == self.array_shapes[3][:-1]
        assert self.h4.image[2:10].shape == self.array_shapes[3][:-1]
        assert self.h4.spectrum[2, 2, 2].shape == self.array_shapes[0]
        assert self.h4.spectrum[2:10, 2:10, 2:10].shape == self.array_shapes[0]

        assert self.h5.image[2].shape == self.array_shapes[4][:-1]
        assert self.h5.image[2:10].shape == self.array_shapes[4][:-1]
        assert self.h5.spectrum[2, 2, 2, 1].shape == self.array_shapes[0]
        assert self.h5.spectrum[2:10, 2:10, 2:10, :1].shape == self.array_shapes[0]

        with pytest.raises(IndexError): _ = self.h2.image[10:20, 10:20]
        with pytest.raises(IndexError): _ = self.h3.image[10:20, 10:20]
        with pytest.raises(IndexError): _ = self.h4.image[10:20, 10:20]
        with pytest.raises(IndexError): _ = self.h5.image[10:20, 10:20]

        with pytest.raises(IndexError): _ = self.h2.spectrum[10:20, 10:20]
        with pytest.raises(IndexError): _ = self.h3.spectrum[10:20, 10:20, 10:20]
        with pytest.raises(IndexError): _ = self.h4.spectrum[10:20, 10:20, 10:20, 10:20]
        with pytest.raises(IndexError): _ = self.h5.spectrum[10:20, 10:20, 10:20, 10:20, 10:20]

    def test_collapse(self):
        with pytest.raises(DimensionError):
            _ = self.h1.collapse()
        for array in (self.h2, self.h3, self.h4, self.h5):
            assert isinstance(array.collapse(), np.ndarray)

        assert self.h2.collapse().shape == self.h2.shape
        assert self.h3.collapse().shape == (100, 30)
        assert self.h4.collapse().shape == (1000, 30)
        assert self.h5.collapse().shape == (2000, 30)

    def test_smoothen(self):
        for array in (self.h1, self.h2, self.h3, self.h4, self.h5):
            assert isinstance(array.smoothen(method='savgol', window_length=3, polyorder=1), hp.hparray)

        assert self.h1.smoothen(method='savgol', window_length=3, polyorder=1).shape == self.h1.shape
        assert self.h2.smoothen(method='savgol', window_length=3, polyorder=1).shape == self.h2.shape
        assert self.h3.smoothen(method='savgol', window_length=3, polyorder=1).shape == self.h3.shape
        assert self.h4.smoothen(method='savgol', window_length=3, polyorder=1).shape == self.h4.shape
        assert self.h5.smoothen(method='savgol', window_length=3, polyorder=1).shape == self.h5.shape

    # testing to ensure that DimensionError is raised if attempting to use plot when number of
    # dimensions is less than 3.
    def test_plot(self):
        # pyqt backend
        with pytest.raises(DimensionError): self.h1.plot()
        with pytest.raises(DimensionError): self.h2.plot()
