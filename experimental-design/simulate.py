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
        dq /= 100*np.sqrt(8*np.log(2))
        probe = QProbe(q, q*dq, data=(r,dr), intensity=scale, background=bkg)
        model = Experiment(probe=probe, sample=sample)

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
        dq /= 100*np.sqrt(8*np.log(2))
        probe = QProbe(q, q*dq, intensity=scale, background=bkg)
        experiment = Experiment(probe=probe, sample=sample)
        _, r_model = experiment.reflectivity()

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
