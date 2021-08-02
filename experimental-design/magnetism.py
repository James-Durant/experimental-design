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

def bayes_factor(times, angle_splits, save_path):
    # Create a .txt file to save the results to.
    file_path = os.path.join(save_path, 'YIG_sample', 'bayes_factor.csv')
    with open(file_path, 'w') as file:
        # Iterate over each time being considered.
        for total_time in times:
            # Define the number of points and counting times for each angle.
            angle_times = [(angle, points, total_time*split)
                           for angle, points, split in angle_splits]
            
            # Simulate data for the YIG sample with a 1 uB/atom magnetic
            # moment in the platinum layer.
            sample = SampleYIG()
            sample.pt_mag.fittable = False
            
            structure = sample.using_conditions()
            structure[3].magnetism.rhoM.value = 0.01638
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
            sampler = Sampler(bumps.fitproblem.FitProblem(experiment))
            logz_1 = sampler.sample(verbose=True, return_evidence=True)
            val_1 = structure[3].magnetism.rhoM.value
            print(val_1)
            
            # Calculate the log-evidence with a 0 uB/atom magnetic
            # moment in the platinum layer.
            structure[3].magnetism.rhoM.value = 0
            experiment = refl1d.experiment.Experiment(sample=structure, probe=probe)
            sampler = Sampler(bumps.fitproblem.FitProblem(experiment))
            logz_2 = sampler.sample(verbose=True, return_evidence=True)
            val_2 = structure[3].magnetism.rhoM.value
            print(val_2)
            
            # Record the Bayes factor between the two models.
            factor = 2*(logz_1-logz_2)
            print(factor)
            print()
            file.write('{0},{1},{2},{3}\n'.format(total_time, factor, val_1, val_2))

def load_bayes():
    file_path = os.path.join(save_path, 'YIG_sample', 'bayes_factor.csv')
    data = np.loadtxt(file_path, delimiter=',')
    times, factors = data[:,0], data[:,1]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(1.5*times, factors)

    ax.set_xlabel('Counting Time (min)', fontsize=11)
    ax.set_ylabel('$\mathregular{Bayes \ Factor \ (2 \ln B_{xy})}$', fontsize=11)

    # Save the plot.
    save_plot(fig, save_path, 'bayes')

def magnetism(yig_thick_range, pt_thick_range, angle_times, save_path, save_views=False):
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

    if save_views:
        save_path = os.path.join(save_path, 'underlayer_choice')
        for i in range(0, 360, 10):
            ax.view_init(elev=40, azim=i)
            save_plot(fig, save_path, 'underlayer_choice_{}'.format(i))

def _func(x, sample, angle_times):
    g = sample.underlayer_info(angle_times, x[0], x[1])
    return -g[0,0]

def optimise(yig_thick_bounds, pt_thick_bounds, angle_times, save_path):
    bounds = [yig_thick_bounds, pt_thick_bounds]

    # Arguments for optimisation function
    args = [SampleYIG(), angle_times]

    # Optimise angles and times, and return results.
    res = differential_evolution(_func, bounds, args=args, polish=False, tol=0.001,
                                 updating='deferred', workers=-1, disp=True)

    return res.x[0], res.x[1], res.fun

def _magnetism_results(save_path='./results'):
    yig_thick_range = np.linspace(400, 900, 75)
    pt_thick_range = np.concatenate((np.linspace(21.5, 30, 50), np.linspace(30, 100, 50)))

    angle_times = [(0.5, 100, 20),
                   (1.0, 100, 40),
                   (2.0, 100, 80)]
    
    #print(-_func((SampleYIG().YIG_thick.value, SampleYIG().Pt_thick.value), SampleYIG(), angle_times))
    #magnetism(yig_thick_range, pt_thick_range, angle_times, save_path, save_views=True)

    yig_thick_bounds = (200, 900)
    pt_thick_bounds = (21.5, 100)
    #yig_thick, pt_thick, val = optimise(yig_thick_bounds, pt_thick_bounds, angle_times, save_path)
    #print('YIG Thickness: {}'.format(yig_thick))
    #print('Pt Thickness: {}'.format(pt_thick))
    #print('Fisher Information: {}'.format(-val))

    times = np.linspace(40, 4000, 100)
    angle_splits = [(0.5, 100, 1/7),
                    (1.0, 100, 2/7),
                    (2.0, 100, 4/7)]
    bayes_factor(times, angle_splits, save_path)
    #plot_bayes()

if __name__ == '__main__':
    _magnetism_results()
