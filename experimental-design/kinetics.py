import matplotlib.pyplot as plt
import numpy as np
import os, sys
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 600
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

from scipy.optimize import differential_evolution

from monolayers import MonolayerDPPG
from utils import save_plot

def _kinetics_results_visualise(save_path):
    """Visualises the choice of measurement angle and contrast SLD for the
       DPPG monolayer model where the lipid area per molecule is degrading
       over time.

    Args:
        save_path (str): path to directory to save results to.

    """
    from monolayers import MonolayerDPPG
    
    # Choose whether the monolayer tailgroups are hydrogenated or deuterated here.
    monolayer = MonolayerDPPG(deuterated=False)
    
    # Angles, contrasts and lipid area per molecule values to consider.
    angle_range = np.linspace(0.2, 4, 75)
    contrast_range = np.linspace(-0.56, 6.36, 75)
    apm_range = np.linspace(54.1039, 500, 20)
    
    # Number of points and measurement time to use when simulating data.
    points = 100
    time = 100
    
    # Iterate over each contrast and angle being considered.
    x, y, infos = [], [], []
    n = len(angle_range)*len(contrast_range) # Number of calculations.
    for i, contrast_sld in enumerate(contrast_range):
        # Display progress.
        if i % 5 == 0:
            print('>>> {0}/{1}'.format(i*len(angle_range), n))

        for angle in angle_range:
            # Record the "true" lipid APM value.
            apm = monolayer.lipid_apm.value
            
            # Split the time budget based on number of apm values.
            angle_times = [(angle, points, time/len(apm_range))]
            
            # Calculate the lipid APM Fisher information for each APM value.
            information = 0
            for new_apm in apm_range:
                # Use the new APM value, corresponding to a degrading sample.
                monolayer.lipid_apm.value = new_apm
                g = monolayer.contrast_info(angle_times, [contrast_sld])
                information += g[0, 0]
            
            monolayer.lipid_apm.value = apm # Reset the lipid APM parameter.
            infos.append(information)
            x.append(contrast_sld)
            y.append(angle)

    # Create the plot of measurment angle and contrast SLD versus Fisher information.
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the surface plot and add a colour bar.
    surface = ax.plot_trisurf(x, y, infos, cmap='plasma')
    fig.colorbar(surface, fraction=0.046, pad=0.04)

    x_label = '$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$'
    y_label = 'Angle (°)'
    z_label = 'Fisher Information'

    ax.set_xlabel(x_label, fontsize=11, weight='bold')
    ax.set_ylabel(y_label, fontsize=11, weight='bold')
    ax.set_zlabel(z_label, fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    # Save the plot.
    save_path = os.path.join(save_path, monolayer.name)
    tg_type = 'deuterated' if monolayer.deuterated else 'hydrogenated'
    filename = 'angle_contrast_choice_' + tg_type
    save_plot(fig, save_path, filename)
    
    # Save different views of the 3D plot.
    # Iterate from 0 to 360 degrees in increments of 10.
    save_path = os.path.join(save_path, 'angle_contrast_choice', tg_type)
    for i in range(0, 360, 10):
        # Set the "camera" view for the 3D plot.
        ax.view_init(elev=40, azim=i)
        
        # Save the view.
        filename_i = filename + '_' + str(i)
        save_plot(fig, save_path, filename_i)

    # Return the contrast and angle pair with greatest Fisher information,
    maximum = np.argmax(infos)
    return x[maximum], y[maximum]

def _kinetics_results_optimise(save_path):
    time = 1000 # A large time improves DE convergence.
    angle_bounds = (0.2, 4.0)
    contrast_bounds = (-0.56, 6.36)
    apm_range = np.linspace(54.1039, 500, 20)
    with open(os.path.join(save_path, monolayer_h.name, 'optimised.txt'), 'w') as file:
        for monolayer, label in zip([monolayer_h, monolayer_d], ['Hydrogenated', 'Deuterated']):
            angle, contrast, val = optimise(monolayer, angle_bounds, contrast_bounds, apm_range, points, time, save_path)

            file.write('-------------- ' + label + ' --------------\n')
            file.write('Angle: {}\n'.format(round(angle, 2)))
            file.write('Contrast: {}\n'.format(round(contrast, 2)))
            file.write('Objective value: {}\n\n'.format(val))

def _func(x, sample, apm_range, points, time):
    """Short summary.

    Args:
        x (type): Description of parameter `x`.
        sample (type): Description of parameter `sample`.
        apm_range (type): Description of parameter `apm_range`.
        points (type): Description of parameter `points`.
        time (type): Description of parameter `time`.

    Returns:
        type: Description of returned object.

    """
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
    """Short summary.

    Args:
        sample (type): Description of parameter `sample`.
        angle_bounds (type): Description of parameter `angle_bounds`.
        contrast_bounds (type): Description of parameter `contrast_bounds`.
        apm_range (type): Description of parameter `apm_range`.
        points (type): Description of parameter `points`.
        time (type): Description of parameter `time`.
        save_path (type): Description of parameter `save_path`.

    Returns:
        type: Description of returned object.

    """
    bounds = [angle_bounds, contrast_bounds]

    # Arguments for optimisation function
    args = [sample, apm_range, points, time]

    # Optimise angles and times, and return results.
    res = differential_evolution(_func, bounds, args=args, polish=False, tol=0.001,
                                 updating='deferred', workers=-1, disp=False)

    return res.x[0], res.x[1], res.fun

if __name__ == '__main__':
    _kinetics_results()
