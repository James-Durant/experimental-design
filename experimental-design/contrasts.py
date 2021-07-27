import matplotlib.pyplot as plt
import numpy as np
import os, sys, time

from optimise import Optimiser
from visualise import contrast_choice_single, contrast_choice_double
from utils import save_plot

# Add the models directory to the system path.
# Add the current directory to the path to avoid issues with threading.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

def _contrast_results_visualise(save_path):
    """Visualises the choice of contrasts for a bilayer sample.

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
    contrast_choice_double(bilayer, contrast_range, angle_times, save_path)

    # Run nested sampling on simulated data.
    bilayer.nested_sampling([6.36, 6.36], angle_times, save_path, 'D2O_D2O')
    bilayer.nested_sampling([-0.56, 6.36], angle_times, save_path, 'H2O_D2O')
    
def _contrast_results_optimise(save_path):
    """Optimises the choice of contrasts for a bilayer sample.

    Args:
        save_path (str): path to directory to save results to.

    """
    from bilayers import BilayerDMPC, BilayerDPPC
    # Choose sample here.
    bilayer = BilayerDMPC()

    # Time budget for experiment.
    total_time = 1000
    
    # Points and proportion of total counting time for each angle.
    angle_splits = [(0.7, 100, 0.2), (2.3, 100, 0.8)]
    
    # Interval containing contrast SLDs to consider.
    contrast_bounds = (-0.56, 6.36)
    
    # Create a new .txt file for the results.
    save_path = os.path.join(save_path, bilayer.name)
    with open(os.path.join(save_path, 'optimised_contrasts.txt'), 'w') as file:
        optimiser = Optimiser(bilayer) # Optimiser for the experiment.
        
        # Optimise the experiment using 1-4 contrasts.
        for i, num_contrasts in enumerate([1, 2, 3, 4]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 4))

            # Time how long the optimisation takes.
            start = time.time()
            results = optimiser.optimise_contrasts(num_contrasts, angle_splits,
                                                   total_time, contrast_bounds,
                                                   verbose=False)
            end = time.time()
            
            contrasts, splits, val = results
            splits = np.array(splits)*100 # Convert to percentages.

            # Round the optimisation function value to 4 significant figures.
            val = np.format_float_positional(val, precision=4, unique=False,
                                             fractional=False, trim='k')

            # Write the conditions, objective value and computation time.
            file.write('----------- {} Contrasts -----------\n'.format(num_contrasts))
            file.write('Contrasts: {}\n'.format(list(np.round(contrasts, 2))))
            file.write('Splits (%): {}\n'.format(list(np.round(splits, 1))))
            file.write('Objective value: {}\n'.format(val))
            file.write('Computation time: {}\n\n'.format(round(end-start, 1)))

def _figure_2(save_path):
    """Creates figure 2 for the paper.

    Args:
        save_path (str): path to directory to save figure to.

    """
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

if __name__ == '__main__':
    save_path = './results'
    _contrast_results_visualise(save_path)
    #_contrast_results_optimise(save_path)
    
    _figure_2('./figures')
