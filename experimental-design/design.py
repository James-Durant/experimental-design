import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 600

from matplotlib.animation import FuncAnimation, PillowWriter

from itertools import combinations
from utils import save_plot

from structures import VariableAngle, VariableContrast, VariableUnderlayer

def angle_choice(sample, initial_angle_times, angle_range, points_new, time_new,
                 save_path, filename, contrasts=[]):
    """Plots the change in minimum eigenvalue of the Fisher information matrix
       as a function of angle choice.

    Args:
        sample (structures.Sample): sample to investigate the angle choice for.
        initial_angle_times (list): points and counting times for each angle already measured.
        angle_range (numpy.ndarray): range of angles to investigate.
        points_new (int): number of points to use for the new data.
        time_new (type): counting time to use for the new data.
        save_path (str): path to directory to save plot to.
        filename (str): filename to use when saving the plot.
        contrasts (list): SLDs of contrasts to simulate, if applicable.

    Returns:
        float: angle with the largest minimum eigenvalue of its Fisher information matrix.

    """
    # Check that the angle can be varied for the sample.
    assert isinstance(sample, VariableAngle)

    # Calculate the information from measurements taken so far.
    g_init = sample.angle_info(initial_angle_times, contrasts)

    min_eigs = []
    # Iterate over each angle to consider.
    for i, angle_new in enumerate(angle_range):
        # Get the information from the new angle.
        new_angle_times = [(angle_new, points_new, time_new)]
        g_new = sample.angle_info(new_angle_times, contrasts)

        # Combine the new information with the existing information.
        min_eigs.append(np.linalg.eigvalsh(g_init+g_new)[0])

        # Display progress.
        if i % 100 == 0:
            print('>>> {0}/{1}'.format(i, len(angle_range)))

    # Plot the minimum eigenvalues vs. angle.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(angle_range, min_eigs)
    ax.set_xlabel('Angle (°)', fontsize=11, weight='bold')
    ax.set_ylabel('Minimum Eigenvalue (arb.)', fontsize=11, weight='bold')

    # Save the plot.
    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'angle_choice_'+filename)

    # Return the angle with largest minimum eigenvalue.
    return angle_range[np.argmax(min_eigs)]

def angle_choice_with_time(sample, initial_angle, angle_range, time_range, points, new_time, save_path, contrasts=[]):
    """Investigates how the second choice of angle for a `sample` changes as the counting time
       of the initial angle is increased.

    Args:
        sample (structures.Sample): sample to investigate the angle choice for.
        initial_angle (float): angle initially measured.
        angle_range (type): range of angles to investigate.
        time_range (type): range of times to investigate.
        points (int): number of points to simulate for each angle.
        new_time (float): counting time for the second angle.
        save_path (str): path to directory to save plot to.
        contrasts (list): SLDs of contrasts to simulate, if applicable.

    """
    # Check that the angle can be varied for the sample.
    assert isinstance(sample, VariableAngle)

    # Create the plot of angle vs. minimum eigenvalue.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Angle (°)', fontsize=11, weight='bold')
    ax.set_ylabel('Minimum Eigenvalue (arb.)', fontsize=11, weight='bold')
    ax.set_xlim(angle_range[0], angle_range[-1])

    # Create the line that will have data added to it.
    line, = ax.plot([], [], lw=3)

    # Initialiser function for the line.
    def init():
        line.set_data([], [])
        return line,

    # Annimation function for the line.
    def animate(i):
        # Get the information from the initial angle at the current time.
        g_init = sample.angle_info([(initial_angle, points, time_range[i])], contrasts)

        min_eigs = []
        # Iterate over each angle to consider.
        for new_angle in angle_range:
            # Combine the information from the first and second angles.
            g_new = sample.angle_info([(new_angle, points, new_time)], contrasts)
            min_eigs.append(np.linalg.eigvalsh(g_init+g_new)[0])
        
        # Update the data of the line.
        ax.set_ylim(min(min_eigs), max(min_eigs))
        line.set_data(angle_range, min_eigs)

        # Display progress.
        if i % 5 == 0:
            print('>>> {0}/{1}'.format(i, len(time_range)))

        return line,

    # Define the animation.
    anim_length = 4000 # Animation length in milliseconds.
    frames = len(time_range) # Number of frames for animation.
    anim = FuncAnimation(fig, animate, init_func=init, blit=True,
                         frames=frames, interval=anim_length//frames)
    plt.close()

    # Save the animation as a gif file.
    writergif = PillowWriter()
    save_path = os.path.join(save_path, sample.name, 'angle_choice_with_time.gif')
    anim.save(save_path, writer=writergif)
    return anim

def contrast_choice_single(sample, contrast_range, initial_contrasts, angle_times, save_path, filename):
    """Plots the change in minimum eigenvalue of the Fisher information matrix
       as a function of a single contrast SLD.

    Args:
        sample (structures.Bilayer): sample to investigate the contrast choice for.
        contrast_range (numpy.ndarray): range of contrast SLDs to investigate.
        initial_contrasts (list): SLDs of contrasts already measured.
        angle_times (list): points and counting time of each angle to simulate.
        save_path (str): path to directory to save plot to.
        filename (str): filename to use when saving the plot.

    Returns:
        float: contrast SLD with the largest minimum eigenvalue of its Fisher information matrix.

    """
    # Check that the contrast can be varied for the sample.
    assert isinstance(sample, VariableContrast)

    # Calculate the information from measurements taken so far.
    g_init = sample.angle_info(angle_times, initial_contrasts)

    min_eigs = []
    # Iterate over each contrast to consider.
    for i, new_contrast in enumerate(contrast_range):
        # Get the information from the new contrast and combine with initial information.
        g_new = sample.contrast_info(angle_times, [new_contrast])
        min_eigs.append(np.linalg.eigvalsh(g_init+g_new)[0])

        # Display progress.
        if i % 100 == 0:
            print('>>> {0}/{1}'.format(i, len(contrast_range)))

    # Plot the minimum eigenvalues vs. contrast SLD.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(contrast_range, min_eigs)
    ax.set_xlabel('$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_ylabel('Minimum Eigenvalue (arb.)', fontsize=11, weight='bold')

    # Save the plot.
    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'contrast_choice_single_'+filename)

    # Return the contrast SLD with largest minimum eigenvalue.
    return contrast_range[np.argmax(min_eigs)]

def contrast_choice_double(sample, contrast_range, angle_times, save_path,
                           reverse_xaxis=True, reverse_yaxis=False):
    """Plots the change in minimum eigenvalue of the Fisher information matrix
       as a function of two contrast SLDs, assuming no prior measurement.

    Args:
        sample (structures.Bilayer): sample to investigate the contrast pair choice for.
        contrast_range (numpy.ndarray): range of contrast SLDs to investigate.
        angle_times (list): points and counting time of each angle to simulate.
        save_path (str): path to directory to save plot to.
        reverse_xaxis (bool): whether to reverse the x-axis.
        reverse_yaxis (bool): whether to reverse the y-axis.

    Returns:
        tuple: contrast SLD pair with the largest minimum eigenvalue.

    """
    # Check that the contrast can be varied for the sample.
    assert isinstance(sample, VariableContrast)

    # Get all possible unique pairs of contrast SLDs.
    contrasts = np.asarray(list(combinations(contrast_range, 2)))

    min_eigs = []
    # Iterate over each contrast pair to consider.
    for i, contrast_pair in enumerate(contrasts):
        # Calculate the minimum eigenvalue of the Fisher information matrix for the pair.
        g = sample.contrast_info(angle_times, contrast_pair)
        min_eigs.append(np.linalg.eigvalsh(g)[0])

        # Display progress.
        if i % 500 == 0:
            print('>>> {0}/{1}'.format(i, len(contrasts)))

    # Duplicate the data so that the plot is not half-empty.
    # This is valid since the choice of contrast is commutative.
    # I.e. the order in which contrasts are measured is not important.
    x = np.concatenate([contrasts[:,0], contrasts[:,1]])
    y = np.concatenate([contrasts[:,1], contrasts[:,0]])
    min_eigs.extend(min_eigs)

    # Create a 3D plot of contrast pair choice vs. minimum eigenvalue.
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, min_eigs, cmap='Spectral')

    # Reverse the x and y axes if specified.
    if reverse_xaxis:
        ax.set_xlim(ax.get_xlim()[::-1])
    if reverse_yaxis:
        ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_xlabel('$\mathregular{Contrast \ 1 \ SLD \ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_ylabel('$\mathregular{Contrast \ 2 \ SLD \ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_zlabel('Minimum Eigenvalue', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    # Save the plot.
    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'contrast_choice_double')

    # Return the contrast SLD pair with largest minimum eigenvalue.
    maximum = np.argmax(min_eigs)
    return x[maximum], y[maximum]

def underlayer_choice(sample, thickness_range, sld_range, contrasts, angle_times, 
                      save_path, filename='', reverse_xaxis=False, reverse_yaxis=False):
    """Plots the change in minimum eigenvalue of the Fisher information matrix
       as a function of a sample's underlayer thickness and SLD.

    Args:
        sample (structures.Bilayer): Description of parameter `sample`.
        thickness_range ((numpy.ndarray): range of underlayer thicknesses to consider.
        sld_range (numpy.ndarray): range of underlayer SLDs to consider.
        contrasts (list): SLDs of contrasts to simulate.
        angle_times (list): points and counting time of each angle to simulate.
        filename
        save_path (str): path to directory to save plot to.
        reverse_xaxis (bool): whether to reverse the x-axis.
        reverse_yaxis (bool): whether to reverse the y-axis.

    Returns:
        tuple: underlayer thickness and SLD with the largest minimum eigenvalue.

    """
    # Check that the underlayer can be varied for the sample.
    assert isinstance(sample, VariableUnderlayer)

    x, y, min_eigs = [], [], []
    # Iterate over each underlayer thickness to investigate.
    for i, thick in enumerate(thickness_range):
        # Iterate over each underlayer SLD to investigate.
        for sld in sld_range:
            # Calculate the minimum eigenvalue of the Fisher information matrix.
            g = sample.underlayer_info(angle_times, contrasts, [(thick, sld)])
            min_eigs.append(np.linalg.eigvalsh(g)[0])
            x.append(thick)
            y.append(sld)

        # Display progress.
        if i % 5 == 0:
            print('>>> {0}/{1}'.format(i*len(sld_range), len(thickness_range)*len(sld_range)))

    # Create a 3D plot of underlayer thickness and SLD vs. minimum eigenvalue.
    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, min_eigs, cmap='Spectral')

    # Reverse the x and y axes if specified.
    if reverse_xaxis:
        ax.set_xlim(ax.get_xlim()[::-1])
    if reverse_yaxis:
        ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_xlabel('$\mathregular{Underlayer\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_ylabel('$\mathregular{Underlayer\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_zlabel('Minimum Eigenvalue', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    # Save the plot.
    save_path = os.path.join(save_path, sample.name)
    filename = 'underlayer_choice' + ('_'+filename if filename != '' else '')
    save_plot(fig, save_path, filename)

    # Return the underlayer thickness and SLD with largest minimum eigenvalue.
    maximum = np.argmax(min_eigs)
    return x[maximum], y[maximum]

def _angle_results(save_path='./results'):
    """Investigates the initial angle choice for a sample and how the choice of
       next angle changes as the counting time of the initial angle increases.

    Args:
        save_path (str): path to directory to save results to.

    """
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import simple_sample, many_param_sample
    from structures import YIG_Sample, SymmetricBilayer, SingleAsymmetricBilayer

    # Choose sample here.
    sample = simple_sample()
    contrasts = []
    #contrasts = [6.36]

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

def _contrast_results(save_path='./results'):
    """Investigates the choice of contrasts for a bilayer sample.

    Args:
        save_path (str): path to directory to save results to.

    """
    from structures import SymmetricBilayer, SingleAsymmetricBilayer

    # Choose sample here.
    bilayer = SingleAsymmetricBilayer()

    # Number of points and counting times for each angle to simulate.
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]

    # Investigate single contrast choices assuming different initial measurements.
    contrast_range = np.linspace(-0.56, 6.36, 500)
    #contrast_choice_single(bilayer, contrast_range, [], angle_times, save_path, 'initial')
    #contrast_choice_single(bilayer, contrast_range, [6.36], angle_times, save_path, 'D2O')
    #contrast_choice_single(bilayer, contrast_range, [-0.56], angle_times, save_path, 'H2O')
    #contrast_choice_single(bilayer, contrast_range, [-0.56, 6.36], angle_times, save_path, 'H2O_D2O')

    # Investigate contrast pair choices assuming no prior measurement.
    contrast_range = np.linspace(-0.55, 6.36, 50)
    #contrast_choice_double(bilayer, contrast_range, angle_times, save_path)

    # Run nested sampling on simulated data to validate the improvements using the suggested designs.
    #bilayer.nested_sampling([6.36, 6.36], angle_times, save_path, 'D2O_D2O', dynamic=False)
    bilayer.nested_sampling([6.36, -0.56], angle_times, save_path, 'D2O_H2O', dynamic=False)

def _underlayer_results(save_path='./results'):
    """Investigates the choice of underlayer thickness and SLD for a bilayer sample.

    Args:
        save_path (str): path to directory to save results to.

    """
    from structures import SymmetricBilayer, SingleAsymmetricBilayer

    # Choose sample here.
    bilayer = SymmetricBilayer()

    # SLDs of contrasts being simulated.
    contrasts = [[6.36], [-0.56], [-0.56, 6.36]]
    # Number of points and counting times for each angle to simulate.
    angle_times = [(0.7, 100, 10), (2.3, 100, 40)]

    # Investigate underlayer choice assuming no prior measurement.
    thickness_range = np.linspace(5, 500, 50)
    sld_range = np.linspace(1, 9, 100)
    
    labels = ['D2O', 'H2O', 'D2O_H2O']
    for c, label in zip(contrasts, labels):
        thick, sld = underlayer_choice(bilayer, thickness_range, sld_range, c, angle_times, save_path, label)
        print('Thickness: {}'.format(round(thick)))
        print('SLD: {}'.format(round(sld, 2)))

if __name__ == '__main__':
    #_angle_results()
    _contrast_results()
    #_underlayer_results()
