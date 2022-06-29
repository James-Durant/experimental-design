import numpy as np
import os

import refnx.reflect
import refl1d.model
import refl1d.probe
import refl1d.experiment


def simulate_magnetic(sample, angle_times, scale=1, bkg=5e-7, dq=2,
                      mm=True, mp=True, pm=True, pp=True, angle_scale=0.7,
                      directbeam_path=None):
    """Simulates an experiment of a given magnetic `sample` measured
       over a number of angles.

    Args:
        sample (refnx.reflect.Stucture or
                refld1d.model.Stack): sample to simulate.
        angle_times (list): points and times for each angle to simulate.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.
        pp (bool): whether to simulate "plus plus" spin state.
        pm (bool): whether to simulate "plus minus" spin state.
        mp (bool): whether to simulate "minus plus" spin state.
        mm (bool): whether to simulate "minus minus" spin state.
        angle_scale (float): angle to use when scaling directbeam flux.
        directbeam_path (str): path to directbeam file to simulate with.

    Returns:
        tuple: model and simulated data for the given `sample`.

    """
    models, datasets = [], []

    # Default path to the directbeam file for OFFSPEC when polarised.
    if directbeam_path is None:
        directbeam_path = os.path.join(os.path.dirname(__file__),
                                       'data',
                                       'directbeams',
                                       'OFFSPEC_polarised_old.dat')
        angle_scale = 0.3

    # Simulate the "minus minus" spin state if requested.
    if mm:
        model, data = simulate(sample, angle_times, scale, bkg, dq,
                               directbeam_path, angle_scale, 0)
        models.append(model)
        datasets.append(data)

    # Simulate the "minus plus" spin state if requested.
    if mp:
        model, data = simulate(sample, angle_times, scale, bkg, dq,
                               directbeam_path, angle_scale, 1)
        models.append(model)
        datasets.append(data)

    # Simulate the "plus minus" spin state if requested.
    if pm:
        model, data = simulate(sample, angle_times, scale, bkg, dq,
                               directbeam_path, angle_scale, 2)
        models.append(model)
        datasets.append(data)

    # Simulate the "plus plus" spin state if requested.
    if pp:
        model, data = simulate(sample, angle_times, scale, bkg, dq,
                               directbeam_path, angle_scale, 3)
        models.append(model)
        datasets.append(data)

    return models, datasets

def simulate(sample, angle_times, scale=1, bkg=5e-6, dq=2, directbeam_path=None,
             angle_scale=0.7, spin_state=None):
    """Simulates an experiment of a `sample` measured over a number of angles.

    Args:
        sample (refnx.reflect.Stucture or
                refld1d.model.Stack): sample to simulate an experiment for.
        angle_times (list): points and times for each angle to simulate.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.
        directbeam_path (str): path to directbeam file to simulate with.
        angle_scale (float): angle to use when scaling directbeam flux.
        spin_state (int): spin state to simulate if given a magnetic sample.

    Returns:
        tuple: model and simulated data for the given `sample`.

    """
    # If there are no angles to measure, return no model and empty data.
    if not angle_times:
        return None, np.zeros((0, 4))

    # Default path to directbeam file for OFFSPEC when non-polarised.
    if directbeam_path is None:
        directbeam_path = os.path.join(os.path.dirname(__file__),
                                       'data',
                                       'directbeams',
                                       'OFFSPEC_non_polarised_old.dat')
        angle_scale = 0.3

    # Iterate over each angle to simulate.
    q, r, dr, counts = [], [], [], []
    total_points = 0
    for angle, points, time in angle_times:
        # Simulate the experiment.
        total_points += points
        simulated = _run_experiment(sample, angle, points, time,
                                    scale, bkg, dq,
                                    directbeam_path, angle_scale, spin_state)

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

    # If there is no data after removing zeros, return no model.
    if len(data) == 0:
        return None, np.zeros((0, 4))

    # If a refnx sample was given, create a refnx ReflectModel.
    if isinstance(sample, refnx.reflect.Structure):
        model = refnx.reflect.ReflectModel(sample, scale=scale, bkg=bkg, dq=dq)

    # If a Refl1D sample was given, create a Refl1D Experiment.
    elif isinstance(sample, refl1d.model.Stack):
        # Create the experiment object.
        q, r, dr = data[:,0], data[:,1], data[:,2]
        model = refl1d_experiment(sample, q, scale, bkg, dq, spin_state)

        # Record the data.
        model.probe.dq = dq
        if sample.ismagnetic:
            model.probe.xs[spin_state].R = r
            model.probe.xs[spin_state].dR = dr
            model.probe.spin_state = spin_state
        else:
            model.probe.R = r
            model.probe.dR = dr

    # Otherwise, the sample must be invalid.
    else:
        raise RuntimeError('invalid sample given')

    return model, data

def _run_experiment(sample, angle, points, time, scale, bkg, dq,
                    directbeam_path, angle_scale, spin_state):
    """Simulates a single angle measurement of a given `sample`.

    Args:
        sample (refnx.reflect.Stucture or
                refld1d.model.Stack): sample to simulate.
        angle (float): angle to simulate.
        points (int): number of points to use for simulated data.
        time (float): counting time for simulation.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.
        directbeam_path (str): path to directbeam file to simulate with.
        angle_scale (float): angle to use when scaling directbeam flux.
        spin_state (int): spin state to simulate if given a magnetic sample.

    Returns:
        tuple: simulated Q, R, dR data and incident neutron counts.

    """
    # Load the directbeam file.
    direct_beam = np.loadtxt(directbeam_path, delimiter=',')
    wavelengths = direct_beam[:,0] # First column is wavelength, second is flux.

    # Adjust flux by measurement angle.
    direct_flux = direct_beam[:,1] * pow(angle/angle_scale, 2)

    # Calculate Q values.
    q = 4*np.pi*np.sin(np.radians(angle)) / wavelengths

    # Bin Q values in equally geometrically-spaced bins using flux as weighting.
    q_bin_edges = np.geomspace(min(q), max(q), points+1)
    flux_binned, _ = np.histogram(q, q_bin_edges, weights=direct_flux)

    # Get the bin centres.
    q_binned = np.asarray([(q_bin_edges[i] + q_bin_edges[i+1]) / 2
                           for i in range(points)])

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

    # Get the measured reflected count for each bin.
    # r_model accounts for background.
    counts_reflected = np.random.poisson(r_model*counts_incident).astype(float)

    # Convert from count space to reflectivity space.
    # Point has zero reflectivity if there is no flux.
    r_noisy = np.divide(counts_reflected, counts_incident,
                        out=np.zeros_like(counts_reflected),
                        where=counts_incident!=0)

    r_error = np.divide(np.sqrt(counts_reflected), counts_incident,
                        out=np.zeros_like(counts_reflected),
                        where=counts_incident!=0)

    return q_binned, r_noisy, r_error, counts_incident

def reflectivity(q, model):
    """Calculates the reflectance of a `model` at given `q` points.

    Args:
        q (numpy.ndarray): Q points to calculate reflectance at.
        model (refnx.reflect.ReflectModel or
               refl1d.experiment.Experiment): model to calculate reflectivity.

    Returns:
        numpy.ndarray: reflectivity for each Q point.

    """
    # If there are no data points, return an empty list.
    if len(q) == 0:
        return []

    # Calculate the reflectance in either refnx or Refl1D.
    if isinstance(model, refnx.reflect.ReflectModel):
        return model(q)

    if isinstance(model, refl1d.experiment.Experiment):
        # If magnetic, use the correct spin state.
        if model.sample.ismagnetic:
            # Get the probe for the spin state to simulate.
            probe = model.probe.xs[model.probe.spin_state]
            scale, bkg, dq = probe.intensity, probe.background, probe.dq

            experiment = refl1d_experiment(model.sample, q,
                                           scale, bkg, dq,
                                           model.probe.spin_state)

            return experiment.reflectivity()[model.probe.spin_state][1]

        # Otherwise, the sample is not magnetic.
        else:
            scale = model.probe.intensity
            bkg = model.probe.background
            dq = model.probe.dq

            experiment = refl1d_experiment(model.sample, q, scale, bkg, dq)
            return experiment.reflectivity()[1]

def refl1d_experiment(sample, q_array, scale, bkg, dq, spin_state=None):
    """Creates a Refl1D experiment for a given `sample` and `q_array`.

    Args:
        sample (structures.Sample): sample to create an experiment for.
        q_array (numpy.ndarray): Q points to use in the experiment.
        scale (float): experimental scale factor.
        bkg (float): level of instrument background noise.
        dq (float): instrument resolution.
        spin_state: spin state to simulate if given a magnetic sample.

    Returns:
        refl1d.experiment.Experiment: experiment for the given `sample`.

    """
    # Transform the resolution from refnx to Refl1D format.
    refl1d_dq = dq / (100*np.sqrt(8*np.log(2)))

    # Calculate the dq array and use it to define a QProbe.
    dq_array = q_array * refl1d_dq
    probe = refl1d.probe.QProbe(q_array, dq_array,
                                intensity=scale, background=bkg)
    probe.dq = dq

    # Adjust probe calculation for constant resolution.
    argmin, argmax = np.argmin(q_array), np.argmax(q_array)
    probe.calc_Qo = np.linspace(q_array[argmin] - 3.5*dq_array[argmin],
                                q_array[argmax] + 3.5*dq_array[argmax],
                                21*len(q_array))

    # If the sample is magnetic, create a polarised QProbe.
    if sample.ismagnetic:
        probes = [None]*4
        probes[spin_state] = probe
        probe = refl1d.probe.PolarizedQProbe(xs=probes, name='')
        probe.spin_state = spin_state

    return refl1d.experiment.Experiment(probe=probe, sample=sample)
