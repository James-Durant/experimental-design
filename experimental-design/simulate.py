import numpy as np

from refnx.reflect import Structure, ReflectModel

from refl1d.model import Stack
from refl1d.probe import QProbe
from refl1d.experiment import Experiment

# Path to directbeam to use for simulation.
DIRECTBEAM_PATH = '../experimental-design/data/directbeam/directbeam_wavelength.dat'

def simulate(sample, angle_times, scale=1, bkg=1e-6, dq=2):
    """Simulates an experiment of a given `sample` measured over a number of angles.

    Args:
        sample (structures.Sample): sample to simulate an experiment for.
        angle_times (list): number of points and counting times for each angle to simulate.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.

    Returns:
        tuple: model and simulated data for the given `sample`.

    """
    q, r, dr, counts = [], [], [], []
    total_points = 0
    # Iterate over each angle to simulate.
    for angle, points, time in angle_times:
        # Simulate the experiment.
        total_points += points
        simulated = run_experiment(sample, angle, points, time, scale, bkg, dq)

        # Combine the data for the angle with the data from previous angles.
        q.append(simulated[0])
        r.append(simulated[1])
        dr.append(simulated[2])
        counts.append(simulated[3])

    # Create a matrix with all of the simulated data.
    data = np.zeros((total_points, 4))
    data[:,0] = np.concatenate(q)
    data[:,1] = np.concatenate(r)
    data[:,2] = np.concatenate(dr)
    data[:,3] = np.concatenate(counts)

    data = data[(data != 0).all(1)] # Remove points of zero reflectivity.
    data = data[data[:,0].argsort()] # Sort by Q.

    # If a refnx sample was given, create a ReflectModel object.
    if isinstance(sample, Structure):
        model = ReflectModel(sample, scale=scale, bkg=bkg, dq=dq)

    # If a Refl1D sample was given, create a Refl1D Experiment object.
    elif isinstance(sample, Stack):
        q, r, dr = data[:,0], data[:,1], data[:,2]
        model = _refl1d_experiment(sample, q, scale, bkg, dq)
        model.data = (r, dr)
        model.dq = dq

    # Otherwise, the given sample must be invalid.
    else:
        raise RuntimeError('invalid sample given')

    return model, data

def run_experiment(sample, angle, points, time, scale, bkg, dq):
    """Simulates a single angle measurement of a given `sample`.

    Args:
        sample (structures.Sample): sample to simulate the experiment on.
        angle (float): measurement angle to simulate.
        points (int): number of data points to use in the simulated data.
        time (float): counting time for the simulation.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.

    Returns:
        tuple: simulated Q, R, dR data with incident neutron counts for each point.

    """
    # Load the directbeam_wavelength.dat file.
    direct_beam = np.loadtxt(DIRECTBEAM_PATH, delimiter=',')
    wavelengths = direct_beam[:,0] # 1st column is wavelength, 2nd is flux.

    # Adjust flux by measurement angle.
    direct_flux = direct_beam[:,1] * pow(angle/0.3, 2)

    # Calculate Q values.
    q = 4*np.pi*np.sin(np.radians(angle)) / wavelengths

    # Bin Q values in equally geometrically-spaced bins using flux as weighting.
    q_bin_edges = np.geomspace(np.min(q), np.max(q), points+1)
    flux_binned, _ = np.histogram(q, q_bin_edges, weights=direct_flux)

    # Get the bin centres.
    q_binned = np.asarray([(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)])

    # Calculate the model reflectivity at each Q point.
    if isinstance(sample, Structure):
        # Create a refnx ReflectModel if the sample was defined in refnx.
        model = ReflectModel(sample, scale=scale, bkg=bkg, dq=dq)
        r_model = model(q_binned)

    elif isinstance(sample, Stack):
        # Create a Refl1D experiment if the sample was defined in Refl1D.
        experiment = _refl1d_experiment(sample, q_binned, scale, bkg, dq)
        r_model = experiment.reflectivity()[1]

    # Otherwise the given sample must be invalid.
    else:
        raise RuntimeError('invalid sample given')

    # Calculate the number of incident neutrons for each bin.
    counts_incident = flux_binned * time

    # Get the measured reflected count for each bin (r_model accounts for background).
    counts_reflected = np.random.poisson(r_model*counts_incident).astype(float)

    # Convert from count space to reflectivity space.
    # Point has zero reflectivity if there is no flux.
    r_noisy = np.divide(counts_reflected, counts_incident,
                        out=np.zeros_like(counts_reflected), where=counts_incident!=0)

    r_error = np.divide(np.sqrt(counts_reflected), counts_incident,
                        out=np.zeros_like(counts_reflected), where=counts_incident!=0)

    return q_binned, r_noisy, r_error, counts_incident

def reflectivity(q, model):
    """Calculates the model reflectivity of a given `model` at a given `q` values.

    Args:
        q (numpy.ndarray): Q values to calculate reflectivity at.
        model (refnx.reflect.ReflectModel or refl1d.experiment.Experiment): model to calculate reflectivity of.

    Returns:
        numpy.ndarray: reflectivity for each Q point.

    """
    # Calculate the reflectivity in either refnx or Refl1D.
    if isinstance(model, ReflectModel):
        return model(q)

    if isinstance(model, Experiment):
        scale, bkg, dq = model.probe.intensity, model.probe.background, model.dq
        experiment = _refl1d_experiment(model.sample, q, scale, bkg, dq)
        return experiment.reflectivity()[1]

def _refl1d_experiment(sample, q_array, scale, bkg, dq):
    """Creates a Refl1D experiment for a given `sample` and `q_array`.

    Args:
        sample (structures.Sample): sample to create an experiment for.
        q_array (numpy.ndarray): Q values to use in the experiment.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.

    Returns:
        refl1d.experiment.Experiment: experiment for given `sample`.

    """
    # Transform the resolution from refnx to Refl1D format.
    dq /= 100*np.sqrt(8*np.log(2))

    # Calculate the dq array and use it to define a Q probe.
    dq_array = q_array*dq
    probe = QProbe(q_array, dq_array, intensity=scale, background=bkg)

    # Adjust probe calculation for constant resolution.
    argmin, argmax = np.argmin(q_array), np.argmax(q_array)
    probe.calc_Qo = np.linspace(q_array[argmin] - 3.5*dq_array[argmin],
                                q_array[argmax] + 3.5*dq_array[argmax],
                                21*len(q_array))

    return Experiment(probe=probe, sample=sample)
