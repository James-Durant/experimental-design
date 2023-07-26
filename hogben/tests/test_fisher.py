import bumps.parameter
import numpy as np
import pytest
import refnx
import refl1d.experiment

from hogben.utils import fisher
from refnx.reflect import SLD as SLD_refnx
from unittest.mock import Mock, patch

Q_VALUES = np.array([[0.1, 0.2, 0.4, 0.6, 0.8]])

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
                           qs=None, step=0.005, importance = None):
    """Obtains the Fisher matrix, and defines the used model parameters"""
    # Provide default values for qs, counts and xi
    if qs is None:
        qs = Q_VALUES
    if counts is None:
        counts = [np.ones(len(qs[0])) * 100]  # Define 100 counts at each q
    if xi is None:
        xi = model.xi
    return fisher(qs, xi, counts, [model], step, importance)

def get_mock_reflectivity():
    """
    Mocks the reflectivity values in the calculation, using two lists. The
    lists are yielded alternating when called.
    """
    r = [[1.0, 0.5, 0.4, 0.2, 0.1], [0.95, 0.45, 0.35, 0.15, 0.05]]
    while True:
        yield r[0]
        yield r[1]

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
    g_reference = get_fisher_information(model, xi=xi)
    np.testing.assert_allclose(g_reference, g_correct, rtol=1e-08)

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
    importance = np.diag([1,2,3])
    g_correct = [
        [1.28125, 1.025, 76.875],
        [0.5125, 0.41, 30.75],
        [25.625, 20.5, 1537.5]
    ]
    g_reference = get_fisher_information(model, xi=xi, importance = importance)
    np.testing.assert_allclose(g_reference, g_correct, rtol=1e-08)