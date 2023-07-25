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

Q_VALUES=np.array([[0.1, 0.2, 0.4, 0.6, 0.8]])

@pytest.fixture
def refl1d_model():
    """Define a bilayer sample, and return the associated refl1d model"""
    # Define sample
    air = SLD_refl1d(rho=0, name='Air')
    layer1 = SLD_refl1d(rho=4, name='Layer 1')(thickness=60, interface=8)
    layer2 = SLD_refl1d(rho=8, name='Layer 2')(thickness=150, interface=2)
    substrate = SLD_refl1d(rho=2.047, name='Substrate')(thickness=0,
                                                        interface=2)
    layer1.thickness.pmp(10)  # Set 10% interval on bounds
    layer2.thickness.pmp(10)
    layer1.interface.pmp(10)
    layer2.interface.pmp(10)
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
    layer1.thick.bounds = (40, 70)
    layer2.thick.bounds = (100, 180)
    layer1.rough.bounds = (6, 10)
    layer2.rough.bounds = (1, 3)
    sample = air | layer1 | layer2 | substrate
    model = refnx.reflect.ReflectModel(sample, scale=1, bkg=5e-6, dq=2)
    # Define model
    model.xi = [layer1.rough, layer2.rough, layer1.thick,
                layer2.thick]  # Default free model parameters
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

def get_fisher_information(model, xi=None, counts=None,
                           qs=None, step=0.005):
    """Obtains the Fisher matrix, and defines the used model parameters"""
    # Provide default values for qs, counts and xi
    if qs is None:
        qs = Q_VALUES  # Define default array of qs
    if counts is None:
        counts = [np.ones(len(qs[0])) * 100]  # Define 100 counts at each q
    if xi is None:
        xi = model.xi
    return fisher(qs, xi, counts, [model], step)

def get_reflectivity_from_datapoints(data_points):
    """
    Mocks the reflectivity values in the calculation, values are changed
    between each call to mimic a changing parameter in the model. Value
    should always be between 0 and 1.
    """
    r = list(np.linspace(1, 0, num=data_points))  # Reflectivity from 1 to 0
    while True:
        r = [abs(value - 0.47) for value in r]  # Change reflectivity after
        # each call
        yield r

def get_mock_reflectivity():
    """
    Mocks the reflectivity values in the calculation, values are changed
    between each call to mimic a changing parameter in the model. Value
    should always be between 0 and 1.
    """
    r = [[1.0, 0.5, 0.4, 0.2, 0.1], [0.95, 0.45, 0.35, 0.15, 0.05]]
    while True:
        yield r[0]
        yield r[1]

def test_fisher_workflow_refnx(refnx_model):
    """
    Tests that all the calculated Fisher information matrix returns the expected values
    for a given set of parameters
    """
    g = get_fisher_information(refnx_model, step=0.005)
    expected_fisher = [
        [1.29426077e-06, 1.12089534e-06, -1.67407318e-07, -9.89857761e-08],
        [1.12089534e-06, 1.00559528e-06, -1.39622503e-07, -7.96457855e-08],
        [-1.67407318e-07, -1.39622503e-07, 2.55843215e-08, 1.71903501e-08],
        [-9.89857761e-08, -7.96457855e-08, 1.71903501e-08, 1.24669272e-08]
    ]
    np.testing.assert_allclose(g, expected_fisher, rtol=1e-08)


def test_fisher_workflow_refl1d(refl1d_model):
    """
    Tests that all the calculated Fisher information matrix returns the
    expected values for a given set of parameters
    """
    g = get_fisher_information(refl1d_model, step=0.005)
    expected_fisher = [
        [7.16085407e-06, 1.29820479e-05, -8.81392856e-07, -5.67164020e-07],
        [1.29820479e-05, 2.44043845e-05, -1.53347962e-06, -9.45044841e-07],
        [-8.81392856e-07, -1.53347962e-06, 1.25317378e-07, 9.12663545e-08],
        [-5.67164020e-07, -9.45044841e-07, 9.12663545e-08, 7.22781693e-08]
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
    g_correct = [[1.28125, 0.5125, 25.625],
                [0.5125, 0.205, 10.25],
                [25.625, 10.25, 512.5]]
    g_reference = get_fisher_information(model, xi = xi)
    np.testing.assert_allclose(g_reference, g_correct, rtol=1e-08)

@pytest.mark.parametrize('model_class', ("refl1d_model",
                                       "refnx_model"))
@pytest.mark.parametrize('step', (0.01, 0.0075, 0.0025, 0.001, 0.0001))
def test_fisher_consistent_steps(step,  model_class, request):
    """
    Tests whether the Fisher information remains mostly consistent when
    changing step size using the refnx model
    """
    model = request.getfixturevalue(model_class)
    g_reference = get_fisher_information(model, step=0.005)
    g_compare = get_fisher_information(model, step=step)
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

    qs = Q_VALUES
    mock_reflectivity.side_effect = get_mock_reflectivity()

    expected_shape = (model_params, model_params)
    g = get_fisher_information(None, xi=xi)
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
    mock_reflectivity.side_effect = get_reflectivity_from_datapoints(len(qs))
    g = get_fisher_information(model, qs=[qs])
    np.testing.assert_array_less(np.zeros(len(g)), np.diag(g))

@pytest.mark.parametrize('model_class', ("mock_refl1d_model",
                                       "mock_refnx_model"))
@pytest.mark.parametrize('model_params', (1, 2, 3, 4))
def test_fisher_no_data(model_params, model_class, request):
    """Tests whether a model with zero data points properly returns a
    zero array"""
    model = request.getfixturevalue(model_class)
    xi = model.xi[:model_params]
    qs = []
    g = get_fisher_information(model, qs=[qs], xi=xi)
    np.testing.assert_equal(g, np.zeros((len(xi), len(xi))))

@pytest.mark.parametrize('model_class', ("mock_refl1d_model",
                                       "mock_refnx_model"))
@patch('hogben.utils.reflectivity')
def test_fisher_no_parameters(mock_reflectivity, model_class, request):
    """Tests whether a model with zero data points properly returns a
    zero array"""
    model = request.getfixturevalue(model_class)
    mock_reflectivity.side_effect = get_mock_reflectivity()
    g = get_fisher_information(model, xi=[])
    np.testing.assert_equal(g.shape, (0,0))
