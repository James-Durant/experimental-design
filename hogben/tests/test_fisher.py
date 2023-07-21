import bumps
import numpy as np
import pytest
import refnx

from hogben.simulate import simulate
from hogben.utils import fisher
from refnx.reflect import SLD as SLD_refnx
from refl1d.material import SLD as SLD_refl1d
from unittest.mock import patch

class MockXi:
    """Mock of Xi, list of varying parameters"""
    def __init__(self, values):
        self._elements = [MockXiElement(value) for value in values]

    def __getitem__(self, index):
        return self._elements[index]

    def __len__(self):
        return len(self._elements)

class MockXiElement:
    """Mock of the elements in Xi, allows to call for the value attribute"""
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        self._value = new_value


@pytest.fixture
def refl1d_model():
    """Define a bilayer sample, and return the associated refnx model"""
    # Define sample
    air = SLD_refl1d(rho=0, name='Air')
    layer1 = SLD_refl1d(rho=4, name='Layer 1')(thickness=60, interface=8)
    layer2 = SLD_refl1d(rho=8, name='Layer 2')(thickness=150, interface=2)
    layer1.thickness.pmp(10)  # Set 10% interval on bounds
    layer2.thickness.pmp(10)
    layer1.interface.pmp(10)
    layer2.interface.pmp(10)
    substrate = SLD_refl1d(rho=2.047, name='Substrate')(thickness=0, interface=2)
    sample = substrate | layer2 | layer1 | air
    # Define model
    angle_times = [(0.7, 100, 5), (2.0, 100, 20)]  # (Angle, Points, Time)
    model, _ = simulate(sample, angle_times)
    model.xi = [layer1.interface, layer2.interface, layer1.thickness,
                layer2.thickness]
    return model


@pytest.fixture
def refnx_model():
    """Define a bilayer sample, and return the associated refl1d model"""
    # Define sample
    air = SLD_refnx(0, name='Air')
    layer1 = SLD_refnx(4, name='Layer 1')(thick=60, rough=8)
    layer2 = SLD_refnx(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD_refnx(2.047, name='Substrate')(thick=0, rough=2)
    layer1.thick.bounds = (40, 70)
    layer2.thick.bounds = (100, 180)
    layer1.rough.bounds = (6, 10)
    layer2.rough.bounds = (1, 3)
    sample = air | layer1 | layer2 | substrate
    model = refnx.reflect.ReflectModel(sample, scale=1, bkg=5e-6, dq=2)
    # Define model
    model.xi = [layer1.rough, layer2.rough, layer1.thick, layer2.thick]  # Default free model parameters
    return model


# @patch('hogben.utils.reflectivity')
def get_fisher_information(model, qs=None, xi=None):
    """Obtains the Fisher matrix, and defines the used model parameters"""
    if qs is None:
        qs = np.array([[0.1, 0.2, 0.4, 0.6, 0.8]])  # Define default array of qs
    counts = [np.ones(len(qs[0])) * 500]  # Define 500 counts at each q point
    if xi is None:
        xi = model.xi
    return fisher(qs, xi, counts, [model])


@patch('hogben.utils.reflectivity')
@patch('hogben.utils._get_bounds')
def get_fisher_mock(mock_bounds, mock_reflectivity, bounds = None, qs = None,
                    xi = None):
    """Obtains a mocked Fisher matrix, without actually calculating the
    reflectivity or model. Allows for custom inputs on the q values,
    parameter values and bounds.
    """
    if qs == None:
        qs = [[0.1, 0.2, 0.4, 0.6, 0.8]]
    if xi is None:
        xi = MockXi([8, 2, 60 , 150])
    counts = [np.ones(len(qs[0])) * 500]  # Define 500 counts at each q point
    mock_reflectivity.side_effect = get_mock_reflectivity(len(qs[0]))
    if bounds is None:
        mock_bounds.return_value = \
            np.array([6.0, 1.0, 40, 100]), np.array([10, 3, 70, 180])
    else:
        mock_bounds.return_value = bounds[0], bounds[1]
    return fisher(qs, xi, counts, [[]])


def get_mock_reflectivity(data_points):
    """
    Mocks the reflectivity values in the calculation, values are changed between each call
    to mimic a changing parameter in the model. Value should always be between 0 and 1.
    """
    if data_points == 0:
        r = []
    else:
        r = list(np.arange(1, 0, -1 / data_points))  # Decaying reflectivity from 1 to 0
    while True:
        r = [abs(value - 0.43) for value in r]  # Change reflectivity after each call
        yield r

class Test_Fisher():
    def test_fisher_information_values_refnx(self, refnx_model):
        """
        Tests that all the calculated Fisher information matrix returns the expected values
        for a given set of parameters
        """
        g = get_fisher_information(refnx_model)
        expected_fisher = [[6.47130383e-06, 5.60447670e-06, -8.37036590e-07, -4.94928881e-07],
                           [5.60447670e-06, 5.02797639e-06, -6.98112513e-07, -3.98228927e-07],
                           [-8.37036590e-07, -6.98112513e-07, 1.27921607e-07, 8.59517503e-08],
                           [-4.94928881e-07, -3.98228927e-07, 8.59517503e-08, 6.23346359e-08]]
        np.testing.assert_allclose(g, expected_fisher, rtol=1e-08, atol=0)

    def test_fisher_information_values_ref1d(self, refl1d_model):
        """
        Tests that all the calculated Fisher information matrix returns the expected values
        for a given set of parameters
        """
        g = get_fisher_information(refl1d_model)
        expected_fisher = [
            [3.58042704e-05, 6.49102395e-05, -4.40696428e-06, -2.83582010e-06],
            [6.49102395e-05, 1.22021923e-04, -7.66739810e-06, -4.72522420e-06],
            [-4.40696428e-06, -7.66739810e-06, 6.26586892e-07, 4.56331772e-07],
            [-2.83582010e-06, -4.72522420e-06, 4.56331772e-07, 3.61390847e-07]]
        np.testing.assert_allclose(g, expected_fisher, rtol=1e-08, atol=0)

    @pytest.mark.parametrize('model_params', (1, 2, 3, 4))
    def test_fisher_shape(self, model_params):
        """
        Tests whether the shape of the Fisher information matrix remains
         correct when changing the amount of parameters
        """
        xi_all = MockXi([8, 2, 60, 150])
        xi = xi_all[:model_params]
        bounds = [np.array([6.0, 1.0, 40, 100][:model_params]),
                  np.array([10, 3, 70, 180][:model_params])]
        expected_shape = (model_params, model_params)
        g = get_fisher_mock(bounds=bounds, xi=xi)
        np.testing.assert_array_equal(g.shape, expected_shape)

    @pytest.mark.parametrize('qs',
                             (np.arange(0.001, 1.0, 0.25),
                              np.arange(0.001, 1.0, 0.10),
                              np.arange(0.001, 1.0, 0.05),
                              np.arange(0.001, 1.0, 0.01)))
    def test_fisher_diagonal_positive(self, qs):
        """Tests whether the diagonal values in the Fisher information matrix
         are positively valued"""
        qs = [0.1, 0.2, 0.4, 0.6, 0.8]
        g = get_fisher_mock(qs=[qs])
        np.testing.assert_array_less(np.zeros(len(g)), np.diag(g))

    @pytest.mark.parametrize('xi',
                             (MockXi([]),
                              MockXi([8]),
                              MockXi([8, 2]),
                              MockXi([8, 2, 60]),
                              MockXi([8, 2, 60, 150])))
    def test_no_data(self, xi):
        """Tests whether a model with zero data points properly returns a
        zero array"""
        qs = []
        g = get_fisher_mock(qs=[qs], xi = xi)
        np.testing.assert_equal(g, np.zeros((len(xi), len(xi))))
