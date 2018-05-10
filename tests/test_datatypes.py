import pytest
import numpy as np

from skhyper.process import Process


class TestDataTypes:
    def test_data_types(self):
        data_empty_list = []
        data_1d = np.random.rand(20)
        data_2d = np.random.rand(2, 20)

        # data type checks
        with pytest.raises(TypeError): Process(data_empty_list)
        with pytest.raises(TypeError): Process(data_1d)
        with pytest.raises(TypeError): Process(data_2d)
