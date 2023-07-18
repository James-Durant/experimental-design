import numpy as np
import pytest

from hogben.simulate import simulate
from hogben.utils import fisher
from refnx.reflect import SLD


@pytest.fixture
def sample():
    air = SLD(0, name='Air')
    layer1 = SLD(4, name='Layer 1')(thick=60, rough=8)
    layer2 = SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    layer1.thick.bounds = (40, 70)
    layer2.thick.bounds = (100, 180)
    layer1.rough.bounds = (6,10)
    layer2.rough.bounds = (1,3)
    return air | layer1 | layer2 | substrate


@pytest.fixture
def model(sample):
    angle_times = [(0.7, 100, 5), (2.0, 100, 20)]  # (Angle, Points, Time)
    model, _ = simulate(sample, angle_times)
    return model


class Test_Fisher():
    qs = [[0.1, 0.2, 0.4, 0.6, 0.8]]
    counts = [[500, 500, 300, 300, 400]]

    @pytest.fixture(autouse=True)
    def get_fisher_information(self, sample, model):
        """Obtains the fisher matrix, and defines the used model parameters"""
        self.xi = [sample[1].thick, sample[2].thick, sample[1].rough, sample[2].rough]
        self.g = fisher(self.qs, self.xi, self.counts, [model])

    def test_elements_greater_than_zero(self):
        """
        Tests that all diagonal elements in the Fisher matrix are greater
        than zero
        """
        np.testing.assert_array_less(np.zeros(len(self.g)), np.diag(self.g))

    def test_fisher_matrix_size(self):
        """
        Tests that the obtained Fisher matrix has the correct size
        """
        matrix_size = [len(self.xi), len(self.xi)]
        g_size = [len(self.g), len(self.g[0])]
        np.testing.assert_equal(matrix_size, g_size)