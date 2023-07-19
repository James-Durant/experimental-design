import numpy as np
import pytest

from hogben.simulate import simulate
from hogben.utils import fisher
from refnx.reflect import SLD
from unittest.mock import patch


@pytest.fixture
def model_fixture():
    """Define a bilayer sample, and return the associated refnx model"""
    # Define sample
    air = SLD(0, name='Air')
    layer1 = SLD(4, name='Layer 1')(thick=60, rough=8)
    layer2 = SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    layer1.thick.bounds = (40, 70)
    layer2.thick.bounds = (100, 180)
    layer1.rough.bounds = (6, 10)
    layer2.rough.bounds = (1, 3)
    sample = air | layer1 | layer2 | substrate

    # Define model
    angle_times = [(0.7, 100, 5), (2.0, 100, 20)]  # (Angle, Points, Time)
    model, _ = simulate(sample, angle_times)
    return model


@patch('hogben.utils.reflectivity')
def get_fisher_information(model_fixture, mock_reflectivity, qs=None, xi=None):
    """Obtains the Fisher matrix, and defines the used model parameters"""
    if qs is None:
        qs = [[0.1, 0.2, 0.4, 0.6, 0.8]]  # Define default array of qs
    counts = [np.ones(len(qs[0])) * 500]  # Define 500 counts at each q point

    layer1 = model_fixture.structure.components[1]
    layer2 = model_fixture.structure.components[2]

    if xi is None:
        xi = [layer1.rough, layer2.rough, layer1.thick, layer2.thick]  # Default free model parameters

    # Mock reflectivity from generator function
    mock_reflectivity.side_effect = get_mock_reflectivity(len(qs[0]))
    return fisher(qs, xi, counts, [model_fixture])


def get_mock_reflectivity(data_points):
    """
    Mocks the reflectivity values in the calculation, values are changed between each call
    to mimic a changing parameter in the model. Value should always be between 0 and 1.
    """
    r = [i / data_points for i in range(data_points+1)]  # Create reflectivity values from 0 to 1
    while True:
        r = [abs(value - 0.43) for value in r]  # Change reflectivity after each call
        yield r


class Test_Fisher():
    def test_fisher_information_values(self, model_fixture):
        """
        Tests that all the calculated Fisher information matrix returns the expected values
        for a given set of parameters
        """
        g = get_fisher_information(model_fixture)
        expected_fisher = [[2.59014803e+04, 2.07211842e+05, 4.60470761e+02, 6.90706141e+01],
                           [2.07211842e+05, 1.65769474e+06, 3.68376609e+03, 5.52564913e+02],
                           [4.60470761e+02, 3.68376609e+03, 8.18614686e+00, 1.22792203e+00],
                           [6.90706141e+01, 5.52564913e+02, 1.22792203e+00, 1.84188304e-01]]
        np.testing.assert_allclose(g, expected_fisher, rtol=1e-08, atol=0)

    @pytest.mark.parametrize('model_params', (1, 2, 3, 4))
    def test_fisher_shape(self, model_fixture, model_params):
        """
        Tests whether the shape of the Fisher information matrix remains correct when changing the
        amount of parameters
        """
        layer1 = model_fixture.structure.components[1]
        layer2 = model_fixture.structure.components[2]
        xi_all = [layer1.rough, layer2.rough, layer1.thick, layer2.thick]
        xi = xi_all[:model_params]
        expected_shape = (len(xi), len(xi))
        g = get_fisher_information(model_fixture, xi=xi)
        np.testing.assert_array_equal(g.shape, expected_shape)

    @pytest.mark.parametrize('qs',
                             ([[i / 1 for i in range(1)]],
                              [[i / 24 for i in range(25)]],
                              [[i / 49 for i in range(50)]],
                              [[i / 149 for i in range(150)]]))
    def test_fisher_diagonal_positive(self, model_fixture, qs):
        """Tests whether the diagonal values in the Fisher information matrix are positively valued"""
        g = get_fisher_information(model_fixture, qs=qs)
        np.testing.assert_array_less(np.zeros(len(g)), np.diag(g))
