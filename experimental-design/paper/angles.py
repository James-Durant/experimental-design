import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
plt.rcParams['figure.dpi'] = 600

from visualise import angle_choice, angle_choice_with_time
from visualise import contrast_choice_single, contrast_choice_double
from visualise import underlayer_choice

from utils import save_plot
    
def _angle_plots(save_path='./results'):
    """Investigates the initial angle choice for a sample and how the choice of
       next angle changes as the counting time of the initial angle increases.

    Args:
        save_path (str): path to directory to save results to.

    """
    from samples import similar_sld_sample_1, similar_sld_sample_2
    from samples import thin_layer_sample_1, thin_layer_sample_2
    from samples import simple_sample, many_param_sample
    from bilayers import BilayerDMPC, BilayerDPPC
    from magnetic import SampleYIG

    # Choose sample here.
    sample = simple_sample()
    
    # Choose contrasts here (only bilayers should use this).
    contrasts = []
    #contrasts = [-0.56, 6.36]

    # Number of points and counting time for the initial angle choice.
    points = 100
    time = 100

    # Get the best angle to initially measure.
    angle_range = np.linspace(0.2, 4, 500)
    initial_angle = angle_choice(sample, [], angle_range, points, time, save_path, 'initial', contrasts)
    print('Initial angle: {}'.format(round(initial_angle, 2)))

    # Investigate how the choice of next angle changes as the counting time of the initial angle is increased.
    angle_range = np.linspace(0.2, 4, 50)
    time_range = np.linspace(0, time*8, 50)
    angle_choice_with_time(sample, initial_angle, angle_range, time_range, points, time, save_path, contrasts)

def _contrast_plots(save_path='./results'):
    """Investigates the choice of contrasts for a sample.

    Args:
        save_path (str): path to directory to save results to.

    """
    from bilayers import BilayerDMPC, BilayerDPPC

    # Choose sample here.
    bilayer = BilayerDMPC()

    # Number of points and counting times for each angle to simulate.
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]

    # Investigate single contrast choices assuming different initial measurements.
    contrast_range = np.linspace(-0.56, 6.36, 500)
    contrast_choice_single(bilayer, contrast_range, [], angle_times, save_path, 'initial')
    contrast_choice_single(bilayer, contrast_range, [6.36], angle_times, save_path, 'D2O')
    contrast_choice_single(bilayer, contrast_range, [-0.56], angle_times, save_path, 'H2O')
    contrast_choice_single(bilayer, contrast_range, [-0.56, 6.36], angle_times, save_path, 'H2O_D2O')

    # Investigate contrast pair choices assuming no prior measurement.
    contrast_range = np.linspace(-0.56, 6.36, 75)
    contrast_choice_double(bilayer, contrast_range, angle_times, save_path, save_views=True)

    # Run nested sampling on simulated data to validate the improvements using the suggested designs.
    bilayer.nested_sampling([6.36, 6.36], angle_times, save_path, 'D2O_D2O')
    bilayer.nested_sampling([-0.56, 6.36], angle_times, save_path, 'H2O_D2O')

def _underlayer_plots(save_path='./results'):
    """Investigates the choice of underlayer thickness and SLD for a sample.

    Args:
        save_path (str): path to directory to save results to.

    """
    from bilayers import BilayerDMPC, BilayerDPPC

    # Choose sample here.
    bilayer = BilayerDMPC()

    # Contrasts to simulate.
    contrasts = [[6.36], [-0.56], [-0.56, 6.36]]
    
    # Number of points and counting times for each angle to simulate.
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]

    # Investigate underlayer choice assuming no prior measurement.
    thickness_range = np.linspace(5, 500, 50)
    sld_range = np.linspace(1, 9, 100)

    labels = ['D2O', 'H2O', 'D2O_H2O']
    for c, label in zip(contrasts, labels):
        thick, sld = underlayer_choice(bilayer, thickness_range, sld_range, c, angle_times, save_path, label, save_views=True)
        print('Thickness: {}'.format(round(thick)))
        print('SLD: {}'.format(round(sld, 2)))

    angle_times = [(0.7, 100, 40)]
    underlayers = [(127.1, 5.39)] # Optimal DMPC bilayer underlayer.
    #underlayers = [(76.5, 9.00)] # Optimal DPPC/Ra LPS bilayer underlayer.
    bilayer.nested_sampling([-0.56, 6.36], angle_times, save_path, 'H2O_without_underlayer', underlayers=[])  
    bilayer.nested_sampling([-0.56, 6.36], angle_times, save_path, 'H2O_with_underlayer', underlayers=underlayers)     

def _figure_2():
    """Creates figure 2 for the paper."""
    from bilayers import BilayerDMPC, BilayerDPPC

    sample_1, sample_2 = BilayerDMPC(), BilayerDPPC()
    
    total_time = 1000
    angle_splits = [(0.7, 100, 0.2), (2.3, 100, 0.8)]

    h2o_split_1 = 0.220
    d2o_split_1 = 0.515
    
    h2o_split_2 = 0.245
    d2o_split_2 = 0.540

    d2o_angle_times_1 = [(angle, points, total_time*split*d2o_split_1) for angle, points, split in angle_splits]
    d2o_angle_times_2 = [(angle, points, total_time*split*d2o_split_2) for angle, points, split in angle_splits]
        
    h2o_angle_times_1 = [(angle, points, total_time*split*h2o_split_1) for angle, points, split in angle_splits]
    h2o_angle_times_2 = [(angle, points, total_time*split*h2o_split_2) for angle, points, split in angle_splits]
    
    nxt_angle_times_1 = [(angle, points, total_time*split*(1-d2o_split_1-h2o_split_1)) for angle, points, split in angle_splits]
    nxt_angle_times_2 = [(angle, points, total_time*split*(1-d2o_split_2-h2o_split_2)) for angle, points, split in angle_splits]

    g_init_1 = sample_1.angle_info(d2o_angle_times_1, [6.36]) + sample_1.angle_info(h2o_angle_times_1, [-0.56])
    g_init_2 = sample_2.angle_info(d2o_angle_times_2, [6.36]) + sample_2.angle_info(h2o_angle_times_2, [-0.56])

    min_eigs_1, min_eigs_2 = [], []
    contrast_range = np.linspace(-0.56, 6.36, 500)
    for i, new_contrast in enumerate(contrast_range):
        g_new_1 = sample_1.contrast_info(nxt_angle_times_1, [new_contrast])
        g_new_2 = sample_2.contrast_info(nxt_angle_times_2, [new_contrast])
        
        min_eigs_1.append(np.linalg.eigvalsh(g_init_1+g_new_1)[0])
        min_eigs_2.append(np.linalg.eigvalsh(g_init_2+g_new_2)[0])

    #print(contrast_range[np.argmax(min_eigs_1)])
    #print(contrast_range[np.argmax(min_eigs_2)])

    fig = plt.figure(figsize=[6,4])
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    line_1 = ax1.plot(contrast_range, min_eigs_1, color='b')
    line_2 = ax2.plot(contrast_range, min_eigs_2, color='g')
    
    ax1.set_xlabel('$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax1.set_ylabel('Minimum Eigenvalue', fontsize=11, weight='bold', color='b')
    ax2.set_ylabel('Minimum Eigenvalue', fontsize=11, weight='bold', color='g')
    ax1.legend(line_1+line_2, ['DMPC', 'DPPC/Ra LPS'], loc=0)

    save_plot(fig, '..', 'figures', 'figure_2')

def _angle_results(optimiser, contrasts, total_time, angle_bounds, save_path='./results'):
    """Optimises the measurement angles and associated counting times for an experiment
       using different numbers of angles.

    Args:
        optimiser (optimise.Optimiser): optimiser for the experiment.
        total_time (float): total time budget for the experiment.
        angle_bounds (tuple): interval containing angles to consider.
        save_path (str): path to directory to save results to.

    """
    save_path = os.path.join(save_path, optimiser.sample.name)

    # Create a new text file for the results.
    with open(os.path.join(save_path, 'optimised_angles.txt'), 'w') as file:
        # Optimise the experiment using 1-4 angles.
        for i, num_angles in enumerate([1, 2, 3, 4]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 4))

            # Time how long the optimisation takes.
            start = time.time()
            angles, splits, val = optimiser.optimise_angle_times(num_angles, contrasts, total_time, angle_bounds, verbose=False)
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


def _contrast_results(optimiser, total_time, angle_splits, contrast_bounds, save_path='./results'):
    """Optimises the contrasts of an experiment using different numbers of contrasts.

    Args:
        optimiser (optimise.Optimiser): optimiser for the experiment.
        total_time (float): time budget for experiment.
        angle_splits (list): points and proportion of total counting time for each angle.
        contrast_bounds (tuple): interval containing contrast SLDs to consider.

        save_path (str): path to directory to save results to.

    """
    save_path = os.path.join(save_path, optimiser.sample.name)

    # Create a new text file for the results.
    with open(os.path.join(save_path, 'optimised_contrasts.txt'), 'w') as file:
        # Optimise the experiment using 1-4 contrasts.
        for i, num_contrasts in enumerate([1, 2, 3, 4]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 4))

            # Time how long the optimisation takes.
            start = time.time()
            contrasts, splits, val = optimiser.optimise_contrasts(num_contrasts, angle_splits, total_time, contrast_bounds, verbose=False)
            end = time.time()
            
            # Convert to percentages.
            splits = np.array(splits)*100

            # Round the optimisation function value to 4 significant figures.
            val = np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')

            # Write the optimised conditions, objective value and computation time to the results file.
            file.write('----------- {} Contrasts -----------\n'.format(num_contrasts))
            file.write('Contrasts: {}\n'.format(list(np.round(contrasts, 2))))
            file.write('Splits (%): {}\n'.format(list(np.round(splits, 1))))
            file.write('Objective value: {}\n'.format(val))
            file.write('Computation time: {}\n\n'.format(round(end-start, 1)))

def _underlayer_results(optimiser, angle_times, contrasts, thick_bounds, sld_bounds, save_path='./results'):
    save_path = os.path.join(save_path, optimiser.sample.name)

    # Create a new text file for the results.
    with open(os.path.join(save_path, 'optimised_underlayers.txt'), 'w') as file:
        g = optimiser.sample.underlayer_info(angle_times, contrasts, [])
        val = -np.linalg.eigvalsh(g)[0]
        val = np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')
        
        file.write('----------- No Underlayers -----------\n')
        file.write('Thicknesses: {}\n'.format([]))
        file.write('SLDs: {}\n'.format([]))
        file.write('Objective value: {}\n\n'.format(val))
        
        # Optimise the experiment using 1-4 contrasts.
        for i, num_underlayers in enumerate([1, 2, 3]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 3))

            # Time how long the optimisation takes.
            start = time.time()
            thicknesses, slds, val = optimiser.optimise_underlayers(num_underlayers, angle_times, contrasts, 
                                                                    thick_bounds, sld_bounds, verbose=False)
            end = time.time()

            # Round the optimisation function value to 4 significant figures.
            val = np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')

            # Write the optimised conditions, objective value and computation time to the results file.
            file.write('----------- {} Underlayers -----------\n'.format(num_underlayers))
            file.write('Thicknesses: {}\n'.format(list(np.round(thicknesses, 1))))
            file.write('SLDs: {}\n'.format(list(np.round(slds, 2))))
            file.write('Objective value: {}\n'.format(val))
            file.write('Computation time: {}\n\n'.format(round(end-start, 1)))

if __name__ == '__main__':
    from bilayers import BilayerDMPC, BilayerDPPC

    sample = BilayerDMPC()
    optimiser = Optimiser(sample)

    total_time = 1000
    angle_splits = [(0.7, 100, 0.2), (2.3, 100, 0.8)]
    contrast_bounds = (-0.56, 6.36)
    _contrast_results(optimiser, total_time, angle_splits, contrast_bounds)

    contrasts = [-0.56, 6.36]
    total_time = 1000
    angle_bounds = (0.2, 4.0)
    _angle_results(optimiser, contrasts, total_time, angle_bounds)
    
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]
    contrasts = [-0.56, 6.36]
    thick_bounds = (0, 500)
    sld_bounds = (1, 9)
    _underlayer_results(optimiser, angle_times, contrasts, thick_bounds, sld_bounds)

if __name__ == '__main__':
    _angle_plots()
    #_contrast_plots()
    #_underlayer_plots()
    #_figure_2()