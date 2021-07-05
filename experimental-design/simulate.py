import numpy as np

from refnx.reflect import Structure, ReflectModel

from refl1d.model import Stack
from refl1d.probe import QProbe
from refl1d.experiment import Experiment

DIRECTBEAM_PATH = '../experimental-design/data/directbeam/directbeam_wavelength.dat'

def simulate(sample, angle_times, scale=1, bkg=1e-6, dq=2):
    q, r, dr, counts = [], [], [], []
    total_points = 0
    for angle, points, time in angle_times:
        # Simulate the experiment for the angle.
        total_points += points
        simulated = run_experiment(sample, angle, points, time, scale, bkg, dq)

        # Combine the data for the angle with the data from other angles.
        q.append(simulated[0])
        r.append(simulated[1])
        dr.append(simulated[2])
        counts.append(simulated[3])

    data = np.zeros((total_points, 4))
    data[:,0] = np.concatenate(q)
    data[:,1] = np.concatenate(r)
    data[:,2] = np.concatenate(dr)
    data[:,3] = np.concatenate(counts)

    data = data[(data != 0).all(1)] # Remove points of zero reflectivity.
    data = data[data[:,0].argsort()] # Sort by Q.

    if isinstance(sample, Structure):
        model = ReflectModel(sample, scale=scale, bkg=bkg, dq=dq)

    elif isinstance(sample, Stack):
        q, r, dr = data[:,0], data[:,1], data[:,2]
        model = _refl1d_experiment(sample, q, dq, scale, bkg)
        model.data = (r, dr)
        model.dq = dq
        
    else:
        raise RuntimeError('invalid sample given')

    return model, data

def run_experiment(sample, angle, points, time, scale, bkg, dq):
    # Load the directbeam_wavelength.dat file.
    direct_beam = np.loadtxt(DIRECTBEAM_PATH, delimiter=',')
    wavelengths = direct_beam[:,0] # 1st column is wavelength, 2nd is flux.

    # Adjust flux by measurement angle.
    direct_flux = direct_beam[:,1] * pow(angle/0.3, 2)

    q = 4*np.pi*np.sin(np.radians(angle)) / wavelengths # Calculate Q values.

    # Bin Q values in equally log-spaced bins using flux as weighting.
    q_bin_edges = np.geomspace(np.min(q), np.max(q), points+1)

    flux_binned, _ = np.histogram(q, q_bin_edges, weights=direct_flux)

    # Get the bin centres and calculate model reflectivity.
    q_binned = np.asarray([(q_bin_edges[i] + q_bin_edges[i+1]) / 2 for i in range(points)])

    if isinstance(sample, Structure):
        model = ReflectModel(sample, scale=scale, bkg=bkg, dq=dq)
        r_model = model(q_binned)

    elif isinstance(sample, Stack):
        experiment = _refl1d_experiment(sample, q_binned, dq, scale, bkg)
        r_model = experiment.reflectivity()[1]

    else:
        raise RuntimeError('invalid sample given')

    # Calculate the number of incident neutrons for each bin.
    counts_incident = flux_binned*time

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
    if isinstance(model, ReflectModel):
        return model(q)

    if isinstance(model, Experiment):
        scale, bkg, dq = model.probe.intensity, model.probe.background, model.dq
        experiment = _refl1d_experiment(model.sample, q, dq, scale, bkg)
        return experiment.reflectivity()[1]

def _refl1d_experiment(sample, q_array, dq, scale, bkg):
    dq /= 100*np.sqrt(8*np.log(2))
    dq_array = q_array*dq
    probe = QProbe(q_array, dq_array, intensity=scale, background=bkg)
    
    argmin, argmax = np.argmin(q_array), np.argmax(q_array)
    probe.calc_Qo = np.linspace(q_array[argmin] - 3.5*dq_array[argmin],
                                q_array[argmax] + 3.5*dq_array[argmax],
                                21*len(q_array))
    
    return Experiment(probe=probe, sample=sample)
