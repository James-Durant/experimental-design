import bumps.parameter
import numpy as np
import pytest
import refnx
import refl1d.experiment

from hogben.simulate import simulate
from hogben.utils import fisher
from refnx.reflect import SLD as SLD_refnx
from refl1d.material import SLD as SLD_refl1d
from unittest.mock import Mock, patch

Q_VALUES = np.array([[0.1, 0.2, 0.4, 0.6, 0.8]])


@pytest.fixture
def refl1d_model():
    """Define a bilayer sample, and return the associated refl1d model"""
    # Define sample
    air = SLD_refl1d(rho=0, name='Air')
    layer1 = SLD_refl1d(rho=4, name='Layer 1')(thickness=60, interface=8)
    layer2 = SLD_refl1d(rho=8, name='Layer 2')(thickness=150, interface=2)
    substrate = SLD_refl1d(rho=2.047, name='Substrate')(thickness=0,
                                                        interface=2)
    layer1.thickness.pm(10)
    layer2.thickness.pm(10)
    layer1.interface.pm(1)
    layer2.interface.pm(1)
    sample = substrate | layer2 | layer1 | air

    # Define model
    angle_times = [(0.7, 100, 5), (2.0, 100, 20)]  # (Angle, Points, Time)
    model, _ = simulate(sample, angle_times)
    model.xi = [layer1.interface, layer2.interface, layer1.thickness,
                layer2.thickness]
    return model


@pytest.fixture
def refnx_model():
    """Define a bilayer sample, and return the associated refnx model"""
    # Define sample
    air = SLD_refnx(0, name='Air')
    layer1 = SLD_refnx(4, name='Layer 1')(thick=60, rough=8)
    layer2 = SLD_refnx(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD_refnx(2.047, name='Substrate')(thick=0, rough=2)
    layer1.thick.bounds = (50, 70)
    layer2.thick.bounds = (140, 160)
    layer1.rough.bounds = (7, 9)
    layer2.rough.bounds = (1, 3)
    sample = air | layer1 | layer2 | substrate
    model = refnx.reflect.ReflectModel(sample, scale=1, bkg=5e-6, dq=2)
    # Define model
    model.xi = [layer1.rough, layer2.rough, layer1.thick, layer2.thick]
    return model


@pytest.fixture
def mock_refnx_model():
    """Define a bilayer sample, and return the associated refnx model"""
    # Define sample
    parameter_values = [(20, 15, 25), (50, 45, 55), (10, 7.5, 8.5), (2, 1.5,
                                                                     2.5)]
    parameters = [
        Mock(spec=refnx.analysis.Parameter, value=value,
             bounds=Mock(lb=lb, ub=ub))
        for value, lb, ub in parameter_values
    ]
    model = Mock(spec=refnx.reflect.ReflectModel, xi=parameters)
    model.xi = parameters
    return model


@pytest.fixture
def mock_refl1d_model():
    """Define a bilayer sample, and return the associated refnx model"""
    # Define sample
    parameter_values = [(20, 15, 25), (50, 45, 55), (10, 7.5, 8.5), (2, 1.5,
                                                                     2.5)]
    parameters = [
        Mock(spec=bumps.parameter.Parameter, value=value,
             bounds=Mock(limits=[lb, ub]))
        for value, lb, ub in parameter_values
    ]
    model = Mock(spec=refl1d.experiment.Experiment, xi=parameters)
    model.xi = parameters
    return model


def get_fisher_information(models, xi=None, counts=None,
                           qs=None, step=0.005):
    """Obtains the Fisher matrix, and defines the used model parameters"""
    # Provide default values for qs, counts and xi
    if qs is None:
        qs = Q_VALUES
    if counts is None:
        counts = [np.ones(len(qs[0])) * 100]  # Define 100 counts at each q.
    if xi is None:
        xi = [] # Concatenate all xi's to a single list of parameters.
        for model in models:
            xi += model.xi
    return fisher(qs, xi, counts, models, step)


def get_reflectivity_given_datapoints(data_points):
    """
    Mocks the reflectivity values in the calculation for an arbitrary
    amount of data points. The value at each data point is changed from the
    initial value by a total of 0.41, mimicking a changing parameter in the
    model. Value should always remain between 0 and 1.
    """
    r = list(np.linspace(1, 0, num=data_points))  # Reflectivity from 1 to 0
    while True:
        r = [abs(value - 0.41) for value in r]  # Change reflectivity after
        # each call
        yield r


def get_mock_reflectivity():
    """
    Mocks the reflectivity values in the calculation, using two lists. The
    lists are yielded alternating when called.
    """
    r = [[1.0, 0.5, 0.4, 0.2, 0.1], [0.95, 0.45, 0.35, 0.15, 0.05]]
    while True:
        yield r[0]
        yield r[1]


def test_fisher_workflow_refnx(refnx_model):
    """
    Runs the entire fisher workflow for the refnx model, and checks that the
    corresponding results are consistent with the expected values
    """
    g = get_fisher_information([refnx_model], step=0.005)
    expected_fisher = [
        [5.17704306e-06, 2.24179068e-06, -5.02221954e-07, -7.91886209e-07],
        [2.24179068e-06, 1.00559528e-06, -2.09433754e-07, -3.18583142e-07],
        [-5.02221954e-07, -2.09433754e-07, 5.75647233e-08, 1.03142100e-07],
        [-7.91886209e-07, -3.18583142e-07, 1.03142100e-07, 1.99470835e-07]
    ]
    np.testing.assert_allclose(g, expected_fisher, rtol=1e-08)


def test_fisher_workflow_refl1d(refl1d_model):
    """
    Runs the entire fisher workflow for the refl1d model, and checks that the
    corresponding results are consistent with the expected values
    """
    g = get_fisher_information([refl1d_model], step=0.005)
    expected_fisher = [
        [4.58294661e-06, 2.07712766e-06, -4.23068571e-07, -6.80596824e-07],
        [2.07712766e-06, 9.76175381e-07, -1.84017555e-07, -2.83513452e-07],
        [-4.23068571e-07, -1.84017555e-07, 4.51142562e-08, 8.21397190e-08],
        [-6.80596824e-07, -2.83513452e-07, 8.21397190e-08, 1.62625881e-07]
    ]
    np.testing.assert_allclose(g, expected_fisher, rtol=1e-08)


@patch('hogben.utils.reflectivity')
@pytest.mark.parametrize('model_class', ("mock_refl1d_model",
                                         "mock_refnx_model"))
def test_fisher_correct_values(mock_reflectivity, model_class, request):
    """
    Tests that the values of the calculated Fisher information matrix
    are calculated correctly.
    """
    model = request.getfixturevalue(model_class)
    xi = model.xi[:3]
    mock_reflectivity.side_effect = get_mock_reflectivity()
    g_correct = [
        [1.28125, 0.5125, 25.625],
        [0.5125, 0.205, 10.25],
        [25.625, 10.25, 512.5]
    ]
    g_reference = get_fisher_information([model], xi=xi)
    np.testing.assert_allclose(g_reference, g_correct, rtol=1e-08)


@pytest.mark.parametrize('model_class', ("refl1d_model",
                                         "refnx_model"))
@pytest.mark.parametrize('step', (0.01, 0.0075, 0.0025, 0.001, 0.0001))
def test_fisher_consistent_steps(step, model_class, request):
    """
    Tests whether the Fisher information remains mostly consistent when
    changing step size using the refnx model
    """
    model = request.getfixturevalue(model_class)
    g_reference = get_fisher_information([model], step=0.005)
    g_compare = get_fisher_information([model], step=step)
    np.testing.assert_allclose(g_reference, g_compare, rtol=1e-02, atol=0)


@patch('hogben.utils.reflectivity')
@pytest.mark.parametrize('model_class', ("mock_refl1d_model",
                                         "mock_refnx_model"))
@pytest.mark.parametrize('model_params', (1, 2, 3, 4))
def test_fisher_shape(mock_reflectivity, model_params, model_class, request):
    """
    Tests whether the shape of the Fisher information matrix remains
     correct when changing the amount of parameters
    """
    model = request.getfixturevalue(model_class)
    xi = model.xi[:model_params]

    mock_reflectivity.side_effect = get_mock_reflectivity()

    expected_shape = (model_params, model_params)
    g = get_fisher_information([model], xi=xi)
    np.testing.assert_array_equal(g.shape, expected_shape)


@patch('hogben.utils.reflectivity')
@pytest.mark.parametrize('model_class', ("mock_refl1d_model",
                                         "mock_refnx_model"))
@pytest.mark.parametrize('qs',
                         (np.arange(0.001, 1.0, 0.25),
                          np.arange(0.001, 1.0, 0.10),
                          np.arange(0.001, 1.0, 0.05),
                          np.arange(0.001, 1.0, 0.01)))
def test_fisher_diagonal_positive(mock_reflectivity, qs, model_class, request):
    """Tests whether the diagonal values in the Fisher information matrix
     are positively valued"""
    model = request.getfixturevalue(model_class)
    mock_reflectivity.side_effect = get_reflectivity_given_datapoints(len(qs))
    g = get_fisher_information([model], qs=[qs])
    np.testing.assert_array_less(np.zeros(len(g)), np.diag(g))


@pytest.mark.parametrize('model_class', ("mock_refl1d_model",
                                         "mock_refnx_model"))
@pytest.mark.parametrize('model_params', (1, 2, 3, 4))
def test_fisher_no_data(model_params, model_class, request):
    """Tests whether a model with zero data points properly returns an empty
    matrix of the correct shape"""
    model = request.getfixturevalue(model_class)
    xi = model.xi[:model_params]
    qs = []
    g = get_fisher_information([model], qs=[qs], xi=xi)
    np.testing.assert_equal(g, np.zeros((len(xi), len(xi))))


@pytest.mark.parametrize('model_class', ("mock_refl1d_model",
                                         "mock_refnx_model"))
@patch('hogben.utils.reflectivity')
def test_fisher_no_parameters(mock_reflectivity, model_class, request):
    """Tests whether a model with zero data points properly returns a
    zero array"""
    model = request.getfixturevalue(model_class)
    mock_reflectivity.side_effect = get_mock_reflectivity()
    g = get_fisher_information([model], xi=[])
    np.testing.assert_equal(g.shape, (0, 0))
