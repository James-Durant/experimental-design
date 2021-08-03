import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (9,7)
plt.rcParams['figure.dpi'] = 600

import numpy as np
import os, sys, time
# Add the models directory to the system path.
# Add the current directory to the path to avoid issues with threading.
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

import refl1d.material, refl1d.magnetism, refl1d.experiment, bumps.fitproblem

from scipy.optimize import differential_evolution

from magnetic import SampleYIG
from optimise import Optimiser
from simulate import simulate_magnetic
from utils import save_plot, Sampler

def _magnetism_results_visualise(save_path):
    """Visualises the choice of measurement angle and contrast SLD for the
       DPPG monolayer model where the lipid area per molecule is decreasing
       over time.

    Args:
        save_path (str): path to directory to save results to.

    """
    yig_thick_range = np.linspace(400, 900, 75)
    pt_thick_range = np.concatenate((np.linspace(21.5, 30, 50), np.linspace(30, 100, 50)))
    
    angle_times = [(0.5, 100, 20),
                   (1.0, 100, 40),
                   (2.0, 100, 80)]
    
    sample = SampleYIG()
    sample.Pt_mag.value = 0.01638

    x, y, infos = [], [], []
    for i, yig_thick in enumerate(yig_thick_range):
        # Display progress.
        if i % 5 == 0:
            print('>>> {0}/{1}'.format(i*len(pt_thick_range), len(pt_thick_range)*len(yig_thick_range)))

        for pt_thick in pt_thick_range:
            g = sample.underlayer_info(angle_times, yig_thick, pt_thick)

            infos.append(g[0,0])
            x.append(yig_thick)
            y.append(pt_thick)

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')

    surface = ax.plot_trisurf(x, y, infos, cmap='plasma')
    fig.colorbar(surface, fraction=0.046, pad=0.04)

    ax.set_xlabel('$\mathregular{YIG\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_ylabel('$\mathregular{Pt\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_zlabel('Fisher Information', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    # Save the plot.
    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'underlayer_choice')

    save_path = os.path.join(save_path, 'underlayer_choice')
    for i in range(0, 360, 10):
        ax.view_init(elev=40, azim=i)
        save_plot(fig, save_path, 'underlayer_choice_{}'.format(i))
        
def _magnetism_results_optimise(save_path):
    """Optimises the choice of measurement angle and contrast SLD for the
       DPPG monolayer model where the lipid area per molecule is decreasing
       over time.

    Args:
        save_path (str): path to directory to save results to.

    """
    sample = SampleYIG()
    
    yig_thick_bounds = (200, 900)
    pt_thick_bounds = (21.1, 100)
    bounds = [yig_thick_bounds, pt_thick_bounds]

    # Arguments for optimisation function
    args = [sample, angle_times]

    file_path = os.path.join(save_path, sample.name, 'optimised_underlayers.txt')
    with open(file_path, 'w') as file:
        # Optimise angle and contrast.
        res = differential_evolution(_optimisation_func, bounds, args=args,
                                     polish=False, tol=0.001,
                                     updating='deferred', workers=-1,
                                     disp=False)

        val = _optimisation_func([None, None], sample, angle_times)
        file.write('-------------- Unoptimised  --------------\n')
        file.write('Fisher Information: {}\n\n'.format(val))

        # Write the angle, contrast and function value to the .txt file.
        file.write('-------------- Optimised --------------\n')
        file.write('YIG Thickness: {}\n'.format(round(angle, 1)))
        file.write('Pt Thickness: {}\n'.format(round(contrast, 1)))
        file.write('Fisher Information: {}\n\n'.format(-res.fun))

def _optimisation_func(x, sample, angle_times):
    g = sample.underlayer_info(angle_times, x[0], x[1])
    return -g[0,0]

def _magnetism_results_logl(save_path):
    """Optimises the choice of measurement angle and contrast SLD for the
       DPPG monolayer model where the lipid area per molecule is decreasing
       over time.

    Args:
        save_path (str): path to directory to save results to.

    """
    times = np.linspace(40, 4000, 150)
    angle_splits = [(0.5, 100, 1/7),
                    (1.0, 100, 2/7),
                    (2.0, 100, 4/7)]
    
    #bayes_factor(26, times, angle_splits, save_path)
    #bayes_factor(80, times, angle_splits, save_path)
    
    save_path = os.path.join(save_path, 'YIG_sample')

    fig = plt.figure(figsize=(6,7))
    ax = fig.add_subplot(111)
    
    for pt_thick in [26, 80]:
        file_path = os.path.join(save_path, 'logl_ratios_{}.csv'.format(pt_thick))
        data = np.loadtxt(file_path, delimiter=',')
        times, factors = data[:,0], data[:,1]
        ax.plot(1.5*times, factors, label='{}Å Pt Thickness'.format(pt_thick))

    ax.set_xlabel('Counting Time (min.)', fontsize=11, weight='bold')
    ax.set_ylabel('Log Likelihood Ratio', fontsize=11, weight='bold')
    ax.legend()

    # Save the plot.
    save_plot(fig, save_path, 'logl_ratios')

def _calc_logl_ratios(pt_thick, times, angle_splits, save_path):
    save_path = os.path.join(save_path, 'YIG_sample')
    file_path = os.path.join(save_path, 'logl_ratios_{}.csv'.format(pt_thick))
    
    # Create a .txt file to save the results to.
    with open(file_path, 'w') as file:
        # Iterate over each time being considered.
        for total_time in times:
            ratios = []
            for _ in range(100):
                # Define the number of points and counting times for each angle.
                angle_times = [(angle, points, split*total_time)
                               for angle, points, split in angle_splits]
                
                # Simulate data for the YIG sample with a 1 uB/atom magnetic
                # moment in the platinum layer.
                sample = SampleYIG()
                sample.pt_mag.value = 0.01638
                
                structure = sample.using_conditions(pt_thick=pt_thick)
                models, datasets = simulate_magnetic(structure, angle_times,
                                                     scale=1, bkg=5e-7, dq=2,
                                                     pp=True, pm=False,
                                                     mp=False, mm=True)
                
                # Extract the probes for the simulated "up" and "down" spin states
                # and combine into a single probe.
                mm = models[0].probe.xs[0]
                pp = models[1].probe.xs[3]
                probe = refl1d.probe.PolarizedQProbe(xs=(mm, None, None, pp), name='')
                
                # Calculate the log-evidence from nested sampling on the simulated
                # data with a model containing the platinum layer magnetic moment.
                experiment = refl1d.experiment.Experiment(sample=structure, probe=probe)
                logl_1 = -experiment.nllf()
                
                sample.pt_mag.value = 0
                experiment = refl1d.experiment.Experiment(sample=structure, probe=probe)
                logl_2 = -experiment.nllf()
                
                ratio = logl_1-logl_2
                ratios.append(ratio)
                
            ratios.sort()
            iqm_ratio = np.mean(ratios[len(ratios)//4:len(ratios)*3//4])
            file.write('{0},{1}\n'.format(total_time, iqm_ratio))
            print(iqm_ratio)

if __name__ == '__main__':
    save_path = './results'
    
    _magnetism_results_visualise(save_path)
    _magnetism_results_optimise(save_path)
    _magnetism_results_logl(save_path)
