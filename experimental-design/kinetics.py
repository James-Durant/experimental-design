import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 600

from scipy.optimize import differential_evolution

from monolayers import MonolayerDPPG
from utils import save_plot

def kinetics(sample, angle_range, contrast_range, apm_range, points, time, save_path,
             save_views=False):
    assert isinstance(sample, MonolayerDPPG)
    
    x, y, infos = [], [], []
    for i, contrast_sld in enumerate(contrast_range):
        for angle in angle_range:
            apm = sample.lipid_apm.value
            angle_times = [(angle, points, time/len(apm_range))]
            information = 0
            
            for new_apm in apm_range:
                sample.lipid_apm.value = new_apm
                information += sample.contrast_info(angle_times, [contrast_sld])[0, 0]

            sample.lipid_apm.value = apm
            infos.append(information)
            x.append(contrast_sld)
            y.append(angle)

        # Display progress.
        if i % 5 == 0:
            print('>>> {0}/{1}'.format((i+1)*len(angle_range), len(angle_range)*len(contrast_range)))

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    
    surface = ax.plot_trisurf(x, y, infos, cmap='plasma')
    fig.colorbar(surface, fraction=0.046, pad=0.04)

    ax.set_xlabel('$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_ylabel('Angle (°)', fontsize=11, weight='bold')
    ax.set_zlabel('Fisher Information', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    # Save the plot.
    save_path = os.path.join(save_path, sample.name)
    tg_type = 'deuterated' if sample.deuterated else 'hydrogenated'
    filename = 'angle_contrast_choice_' + tg_type
    save_plot(fig, save_path, filename)

    if save_views:
        save_path = os.path.join(save_path, 'angle_contrast_choice', tg_type)
        for i in range(0, 360, 10):
            ax.view_init(elev=40, azim=i)
            
            filename_i = filename + '_' + str(i)
            save_plot(fig, save_path, filename_i)

def _func(x, sample, apm_range, points, time):
    angle, contrast_sld = x[0], x[1]
    angle_times = [(angle, points, time/len(apm_range))]
    
    information = 0
    apm = sample.lipid_apm.value
    for new_apm in apm_range:
        sample.lipid_apm.value = new_apm
        information += sample.contrast_info(angle_times, [contrast_sld])[0, 0]

    sample.lipid_apm.value = apm
    return -information

def optimise(sample, angle_bounds, contrast_bounds, apm_range, points, time, save_path):
    bounds = [angle_bounds, contrast_bounds]

    # Arguments for optimisation function
    args = [sample, apm_range, points, time]
    
    # Optimise angles and times, and return results.
    res = differential_evolution(_func, bounds, args=args, polish=False, tol=0.001,
                                 updating='deferred', workers=-1, disp=False)
    
    return res.x[0], res.x[1], res.fun

def _kinetics_results(save_path='./results'):
    angle_range = np.linspace(0.2, 4, 75)
    contrast_range = np.linspace(-0.56, 6.36, 75)
    apm_range = np.linspace(54.1039, 500, 20)
    points = 100
    time = 100
    
    monolayer_h, monolayer_d = MonolayerDPPG(deuterated=False), MonolayerDPPG(deuterated=True)
    
    #kinetics(monolayer_h, angle_range, contrast_range, apm_range, points, time, save_path, save_views=True)
    #kinetics(monolayer_d, angle_range, contrast_range, apm_range, points, time, save_path, save_views=True)
    
    time = 1000 # A large time improves DE convergence.
    angle_bounds = (0.2, 4.0)
    contrast_bounds = (-0.56, 6.36)
    with open(os.path.join(save_path, monolayer_h.name, 'optimised.txt'), 'w') as file:
        for monolayer, label in zip([monolayer_h, monolayer_d], ['Hydrogenated', 'Deuterated']):
            angle, contrast, val = optimise(monolayer, angle_bounds, contrast_bounds, apm_range, points, time, save_path)
    
            file.write('-------------- ' + label + ' --------------\n')
            file.write('Angle: {}\n'.format(round(angle, 2)))
            file.write('Contrast: {}\n'.format(round(contrast, 2)))
            file.write('Objective value: {}\n\n'.format(val))

if __name__ == '__main__':
    _kinetics_results()
