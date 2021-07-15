import numpy as np

import refnx.reflect
import refl1d.model, refl1d.probe, refl1d.experiment

# Path to directbeam to use for simulation.
DIRECTBEAM_PATH = '../experimental-design/data/directbeam/directbeam_wavelength.dat'

def simulate_magnetic(sample, angle_times, scale=1, bkg=1e-6, dq=2,
                      pp=True, pm=True, mp=True, mm=True):
    models, datasets = [], []

    if mm:
        model, data = simulate(sample, angle_times, scale, bkg, dq, spin_state=0)
        models.append(model)
        datasets.append(data)
        
    if mp:
        model, data = simulate(sample, angle_times, scale, bkg, dq, spin_state=1)
        models.append(model)
        datasets.append(data)
        
    if pm:
        model, data = simulate(sample, angle_times, scale, bkg, dq, spin_state=2)
        models.append(model)
        datasets.append(data)
        
    if pp:
        model, data = simulate(sample, angle_times, scale, bkg, dq, spin_state=3)
        models.append(model)
        datasets.append(data)
        
    return models, datasets

def simulate(sample, angle_times, scale=1, bkg=1e-6, dq=2, spin_state=None):
    """Simulates an experiment of a given `sample` measured over a number of angles.

    Args:
        sample (refnx.reflect.Stucture or refld1d.model.Stack): sample to simulate an experiment for.
        angle_times (list): number of points and counting times for each angle to simulate.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.
        spin_state

    Returns:
        tuple: model and simulated data for given `sample`.

    """
    if not angle_times:
        return None, np.zeros((0, 4))
    
    q, r, dr, counts = [], [], [], []
    total_points = 0
    # Iterate over each angle to simulate.
    for angle, points, time in angle_times:
        # Simulate the experiment.
        total_points += points
        simulated = _run_experiment(sample, angle, points, time, scale, bkg, dq, spin_state)

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

    # If a refnx sample was given, create a ReflectModel.
    if isinstance(sample, refnx.reflect.Structure):
        model = refnx.reflect.ReflectModel(sample, scale=scale, bkg=bkg, dq=dq)

    # If a Refl1D sample was given, create a Refl1D Experiment.
    elif isinstance(sample, refl1d.model.Stack):
        q, r, dr = data[:,0], data[:,1], data[:,2]
        model = refl1d_experiment(sample, q, scale, bkg, dq, spin_state)
        model.probe.xs[spin_state].R = r
        model.probe.xs[spin_state].dR = dr
        model.probe.dq = dq
        model.probe.spin_state = spin_state

    # Otherwise, the sample must be invalid.
    else:
        raise RuntimeError('invalid sample given')

    return model, data

def _run_experiment(sample, angle, points, time, scale, bkg, dq, spin_state, q_min=0.005, q_max=0.3):
    """Simulates a single angle measurement of a given `sample`.

    Args:
        sample (refnx.reflect.Stucture or refld1d.model.Stack): sample to simulate the experiment on.
        angle (float): measurement angle to simulate.
        points (int): number of data points to use in simulated data.
        time (float): counting time for simulation.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.
        spin_state

    Returns:
        tuple: simulated Q, R, dR data and incident neutron counts for each point.

    """
    # Load the directbeam_wavelength.dat file.
    direct_beam = np.loadtxt(DIRECTBEAM_PATH, delimiter=',')
    wavelengths = direct_beam[:,0] # 1st column is wavelength, 2nd is flux.

    # Adjust flux by measurement angle.
    direct_flux = direct_beam[:,1] * pow(angle/0.3, 2)

    # Calculate Q values.
    q = 4*np.pi*np.sin(np.radians(angle)) / wavelengths

    # Bin Q values in equally geometrically-spaced bins using flux as weighting.
    q_bin_edges = np.geomspace(q_min, q_max, points+1)
    flux_binned, _ = np.histogram(q, q_bin_edges, weights=direct_flux)

    # Get the bin centres.
    q_binned = np.asarray([(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)])

    # Calculate the model reflectivity at each Q point.
    if isinstance(sample, refnx.reflect.Structure):
        # Create a refnx ReflectModel if the sample was defined in refnx.
        model = refnx.reflect.ReflectModel(sample, scale=scale, bkg=bkg, dq=dq)
        r_model = model(q_binned)

    elif isinstance(sample, refl1d.model.Stack):
        # Create a Refl1D experiment if the sample was defined in Refl1D.
        experiment = refl1d_experiment(sample, q_binned, scale, bkg, dq, spin_state)
        r_model = reflectivity(q_binned, experiment)

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
    """Calculates the model reflectivity of a given `model` at `q`.

    Args:
        q (numpy.ndarray): Q values to calculate reflectivity at.
        model (refnx.reflect.ReflectModel or refl1d.experiment.Experiment): model to calculate the reflectivity of.

    Returns:
        numpy.ndarray: reflectivity for each Q point.

    """
    if len(q) == 0:
        return []
    
    # Calculate the reflectivity in either refnx or Refl1D.
    if isinstance(model, refnx.reflect.ReflectModel):
        return model(q)

    if isinstance(model, refl1d.experiment.Experiment):
        if model.sample.ismagnetic:
            probe = model.probe.xs[model.probe.spin_state]
            scale, bkg, dq = probe.intensity, probe.background, probe.dq
            experiment = refl1d_experiment(model.sample, q, scale, bkg, dq, model.probe.spin_state)
            return experiment.reflectivity()[model.probe.spin_state][1]
        
        else:
            scale, bkg, dq = model.probe.intensity, model.probe.background, model.probe.dq
            experiment = refl1d_experiment(model.sample, q, scale, bkg, dq)
            return experiment.reflectivity()[1]

def refl1d_experiment(sample, q_array, scale, bkg, dq, spin_state=None):
    """Creates a Refl1D experiment for a given `sample` and `q_array`.

    Args:
        sample (structures.Sample): sample to create an experiment for.
        q_array (numpy.ndarray): Q values to use in the experiment.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.
        spin_state

    Returns:
        refl1d.experiment.Experiment: experiment for given `sample`.

    """
    # Transform the resolution from refnx to Refl1D format.
    refl1d_dq = dq / (100*np.sqrt(8*np.log(2)))

    # Calculate the dq array and use it to define a Q probe.
    dq_array = q_array * refl1d_dq
    probe = refl1d.probe.QProbe(q_array, dq_array, intensity=scale, background=bkg)
    probe.dq = dq

    # Adjust probe calculation for constant resolution.
    argmin, argmax = np.argmin(q_array), np.argmax(q_array)
    probe.calc_Qo = np.linspace(q_array[argmin] - 3.5*dq_array[argmin],
                                q_array[argmax] + 3.5*dq_array[argmax],
                                21*len(q_array))
    
    if sample.ismagnetic:
        probes = [None]*4
        probes[spin_state] = probe
        probe = refl1d.probe.PolarizedQProbe(xs=probes, name='')
        probe.spin_state = spin_state

    return refl1d.experiment.Experiment(probe=probe, sample=sample)
