import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__)))
plt.rcParams['figure.dpi'] = 600

import refl1d.material, refl1d.magnetism, refl1d.experiment, bumps.fitproblem

from scipy.optimize import differential_evolution

from optimise import Optimiser
from magnetic import SampleYIG
from simulate import simulate_magnetic
from utils import save_plot, Sampler

def test(sample, pt_mag):
    pt_magnetism = refl1d.magnetism.Magnetism(rhoM=pt_mag, thetaM=sample.mag_angle)
    yig_magnetism = refl1d.magnetism.Magnetism(rhoM=sample.YIG_mag, thetaM=sample.mag_angle)
    
    air = refl1d.material.SLD(rho=0, name='Air')
    Pt = refl1d.material.SLD(rho=sample.Pt_sld, name='Pt')(sample.Pt_thick, sample.Pt_rough, magnetism=pt_magnetism)
    FePt = refl1d.material.SLD(rho=sample.FePt_sld, name='FePt')(sample.FePt_thick, sample.FePt_rough)
    YIG = refl1d.material.SLD(rho=sample.YIG_sld, name='YIG')(sample.YIG_thick, sample.YIG_rough, magnetism=yig_magnetism)
    sub = refl1d.material.SLD(rho=sample.sub_sld, name='Substrate')(0, sample.sub_rough)
    
    return sub | YIG | FePt | Pt | air

def bayes(times, angle_splits, save_path):
    with open(os.path.join(save_path, 'YIG_sample', 'bayes.csv'), 'w') as file:
        for time in times:
            angle_times = [(angle, points, time*split) for angle, points, split in angle_splits]
            
            sample = SampleYIG(vary=False)
            sample.Pt_thick.range(0, 0.2)
            models, datasets = simulate_magnetic(test(sample, 0.01638), angle_times, scale=1, bkg=5e-7, dq=2,
                                                 pp=True, pm=False, mp=False, mm=True)
        
            mm = models[0].probe.xs[0]
            pp = models[1].probe.xs[3]
        
            probe = refl1d.probe.PolarizedQProbe(xs=(mm, None, None, pp), name='Probe')
            experiment = refl1d.experiment.Experiment(sample=sample.structure, probe=probe)
            sampler = Sampler(bumps.fitproblem.FitProblem(experiment))
            logz_1 = sampler.sample(verbose=False, return_evidence=True)
            
            sample = SampleYIG(vary=False)
            sample.Pt_mag.value = 0
            sample.Pt_thick.range(0, 0.2)
            
            experiment = refl1d.experiment.Experiment(sample=sample.structure, probe=probe)
            sampler = Sampler(bumps.fitproblem.FitProblem(experiment))
            logz_2 = sampler.sample(verbose=False, return_evidence=True)
            
            factor = 2*(logz_1-logz_2)
            print(factor)
            file.write('{0},{1}\n'.format(time, factor))

def plot_bayes(save_path='./results/YIG_sample'): 
    data = np.loadtxt(os.path.join(save_path, 'bayes.csv'), delimiter=',')
    times, factors = data[:,0], data[:,1]
    
    fig = plt.figure(figsize=[9,7])
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
    
def _optimise_angles(total_time, angle_bounds, save_path='./results/YIG_sample'):
    sample = SampleYIG()
    sample.params = [sample.Pt_thick]
    optimiser = Optimiser(sample)

    # Create a new text file for the results.
    with open(os.path.join(save_path, 'optimised_angles.txt'), 'w') as file:
        # Optimise the experiment using 1-4 angles.
        for i, num_angles in enumerate([1, 2, 3, 4]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 4))

            # Time how long the optimisation takes.
            start = time.time()
            angles, splits, val = optimiser.optimise_angle_times(num_angles, None, total_time, angle_bounds, verbose=False)
            end = time.time()

            # Convert to percentages.
            splits = np.array(splits)*100

            # Round the optimisation function value to 4 significant figures.
            val = np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')

            # Write the optimised conditions, objective value and computation time to the results file.
            file.write('----------- {} Angles -----------\n'.format(num_angles))
            file.write('Angles: {}\n'.format(list(np.round(angles, 2))))
            file.write('Splits (%): {}\n'.format(list(np.round(splits, 1))))
            file.write('Objective value: {}\n'.format(val))
            file.write('Computation time: {}\n\n'.format(round(end-start, 1)))
    
def _magnetism_results(save_path='./results'):
    
    total_time = 1000
    angle_bounds = (0.2, 4.0)
    _optimise_angles(total_time, angle_bounds)
    
    
    yig_thick_range = np.linspace(400, 900, 75)
    pt_thick_range = np.concatenate((np.linspace(21.5, 30, 50), np.linspace(30, 100, 50)))
    
    angle_times = [(0.5, 100, 20),
                   (1.0, 100, 40),
                   (2.0, 100, 80)]
    print(-_func((SampleYIG().YIG_thick.value, SampleYIG().Pt_thick.value), SampleYIG(), angle_times))
    #magnetism(yig_thick_range, pt_thick_range, angle_times, save_path, save_views=True)
    
    yig_thick_bounds = (200, 900)
    pt_thick_bounds = (21.5, 100)
    #yig_thick, pt_thick, val = optimise(yig_thick_bounds, pt_thick_bounds, angle_times, save_path)
    #print('YIG Thickness: {}'.format(yig_thick))
    #print('Pt Thickness: {}'.format(pt_thick))
    #print('Fisher Information: {}'.format(-val))
    
    times = np.linspace(20, 1440, 25)
    angle_splits = [(0.5, 100, 1/7),
                    (1.0, 100, 2/7),
                    (2.0, 100, 4/7)]
    #bayes(times, angle_splits, save_path)
    #plot_bayes()
    
if __name__ == '__main__':
    _magnetism_results()