import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['figure.dpi'] = 600

from matplotlib.animation import FuncAnimation, PillowWriter

from itertools import combinations
from utils import fisher, save_plot

from structures import VariableAngle, VariableContrast, VariableUnderlayer

def angle_choice(sample, initial_angle_times, angle_range, points_new, time_new,
                 save_path, filename, contrasts=[]):
    assert isinstance(sample, VariableAngle)

    xi = sample.parameters
    min_eigs = []
    for i, angle_new in enumerate(angle_range):
        new_angle_times = initial_angle_times + [(angle_new, points_new, time_new)]
        qs, counts, models = sample.angle_info(new_angle_times, contrasts)
        
        g = fisher(qs, xi, counts, models)
        min_eigs.append(np.linalg.eigvalsh(g)[0])

        if i % 100 == 0:
            print('>>> {0}/{1}'.format(i, len(angle_range)))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(angle_range, min_eigs)

    ax.set_xlabel('Angle (°)', fontsize=11, weight='bold')
    ax.set_ylabel('Minimum Eigenvalue (arb.)', fontsize=11, weight='bold')

    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'angle_choice_'+filename)

    return angle_range[np.argmax(min_eigs)]

def angle_choice_with_time(sample, initial_angle, angle_range, time_range, points, new_time, save_path, contrasts=[]):
    assert isinstance(sample, VariableAngle)

    xi = sample.parameters
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_xlim(angle_range[0], angle_range[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Angle (°)', fontsize=11, weight='bold')
    ax.set_ylabel('Minimum Eigenvalue (arb.)', fontsize=11, weight='bold')
    
    line, = ax.plot([], [], lw=3)
    
    def init():
        line.set_data([], [])
        return line,
    
    def animate(i):
        min_eigs = []
        for new_angle in angle_range:
            angle_times = [(initial_angle, points, time_range[i]),
                           (new_angle, points, new_time)]
            
            qs, counts, models = sample.angle_info(angle_times, contrasts)
    
            g = fisher(qs, xi, counts, models)
            min_eigs.append(np.linalg.eigvalsh(g)[0])

        min_eigs = np.asarray(min_eigs)
        min_eigs = (min_eigs-min(min_eigs)) / (max(min_eigs)-min(min_eigs))
        
        line.set_data(angle_range, min_eigs)

        if i % 5 == 0:
            print('>>> {0}/{1}'.format(i, len(time_range)))
            
        return line,

    anim_length = 4000 # miliseconds
    frames = len(time_range)
    anim = FuncAnimation(fig, animate, init_func=init, blit=True,
                         frames=frames, interval=anim_length//frames)
    
    plt.close()
    writergif = PillowWriter() 
    save_path = os.path.join(save_path, sample.name, 'angle_choice_with_time.gif')
    anim.save(save_path, writer=writergif)
    return anim

def contrast_choice_single(sample, contrast_range, contrasts, angle_times, save_path, filename):
    assert isinstance(sample, VariableContrast)

    xi = sample.parameters
    qs_init, counts_init, models_init = sample.contrast_info(angle_times, contrasts)

    min_eigs = []
    for i, contrast in enumerate(contrast_range):
        qs_new, counts_new, models_new = sample.contrast_info(angle_times, [contrast])

        qs = qs_init + qs_new
        counts = counts_init + counts_new
        models = models_init + models_new

        g = fisher(qs, xi, counts, models)
        min_eigs.append(np.linalg.eigvalsh(g)[0])

        # Display progress.
        if i % 100 == 0:
            print('>>> {0}/{1}'.format(i, len(contrast_range)))

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(contrast_range, min_eigs)

    ax.set_xlabel('$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_ylabel('Minimum Eigenvalue (arb.)', fontsize=11, weight='bold')

    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'contrast_choice_single_'+filename)

    return contrast_range[np.argmax(min_eigs)]

def contrast_choice_double(sample, contrast_range, angle_times, save_path,
                           reverse_xaxis=True, reverse_yaxis=True):
    assert isinstance(sample, VariableContrast)

    xi = sample.parameters
    contrasts = np.asarray(list(combinations(contrast_range, 2)))

    min_eigs = []
    for i, contrast_pair in enumerate(contrasts):
        qs, counts, models = sample.contrast_info(angle_times, contrast_pair)

        g = fisher(qs, xi, counts, models)
        min_eigs.append(np.linalg.eigvalsh(g)[0])

        # Display progress.
        if i % 500 == 0:
            print('>>> {0}/{1}'.format(i, len(contrasts)))

    x = np.concatenate([contrasts[:,0], contrasts[:,1]])
    y = np.concatenate([contrasts[:,1], contrasts[:,0]])

    min_eigs.extend(min_eigs)

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, min_eigs, cmap='Spectral')

    if reverse_xaxis:
        ax.set_xlim(ax.get_xlim()[::-1])
    
    if reverse_yaxis:
        ax.set_ylim(ax.get_ylim()[::-1])
        
    ax.set_xlabel('$\mathregular{Contrast \ 1 \ SLD \ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_ylabel('$\mathregular{Contrast \ 2 \ SLD \ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_zlabel('Minimum Eigenvalue (arb.)', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'contrast_choice_double')

    maximum = np.argmax(min_eigs)
    return x[maximum], y[maximum]

def underlayer_choice(sample, thickness_range, sld_range, contrasts, angle_times, save_path,
                      reverse_xaxis=False, reverse_yaxis=False):
    assert isinstance(sample, VariableUnderlayer)

    xi = sample.parameters
    x, y, min_eigs = [], [], []
    for i, thick in enumerate(thickness_range):
        for sld in sld_range:
            qs, counts, models = sample.underlayer_info(angle_times, contrasts, (thick, sld))
            x.append(thick)
            y.append(sld)

            g = fisher(qs, xi, counts, models)
            min_eigs.append(np.linalg.eigvalsh(g)[0])

        if i % 5 == 0:
            print('>>> {0}/{1}'.format(i*len(sld_range), len(thickness_range)*len(sld_range)))

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, min_eigs, cmap='Spectral')

    if reverse_xaxis:
        ax.set_xlim(ax.get_xlim()[::-1])
    
    if reverse_yaxis:
        ax.set_ylim(ax.get_ylim()[::-1])

    ax.set_ylabel('$\mathregular{Underlayer\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_xlabel('$\mathregular{Underlayer\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_zlabel('Minimum Eigenvalue', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'underlayer_choice')

    maximum = np.argmax(min_eigs)
    return x[maximum], y[maximum]

def angle_results_normal(save_path):
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import simple_sample, many_param_sample

    sample = thin_layer_sample_1()

    points = 150
    time = 40

    angle_range = np.linspace(0.2, 2.3, 500)
    initial_angle = angle_choice(sample, [], angle_range, points, time, save_path, 'initial')
    print('Initial angle: {}'.format(round(initial_angle, 2)))

    angle_range = np.linspace(0.2, 2.3, 50)
    time_range = np.linspace(0, time*10, 50)
    angle_choice_with_time(sample, initial_angle, angle_range, time_range, points, time, save_path)

def angle_results_bilayer(save_path):
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    sample = SymmetricBilayer()

    contrasts = [6.36]
    points = 150
    time = 200
    
    angle_range = np.linspace(0.2, 2.3, 500)
    initial_angle = angle_choice(sample, [], angle_range, points, time, save_path, 'initial', contrasts)
    print('Initial angle: {}'.format(round(initial_angle, 2)))
    
    angle_range = np.linspace(0.2, 2.3, 50)
    time_range = np.linspace(0, time*8, 50)
    angle_choice_with_time(sample, initial_angle, angle_range, time_range, points, time, save_path, contrasts)

def contrast_results(save_path):
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    bilayer = SymmetricBilayer()
    angle_times = [(0.5, 70, 10), (2.3, 70, 40)]

    contrast_range = np.linspace(-0.55, 6.36, 500)
    contrast_choice_single(bilayer, contrast_range, [], angle_times, save_path, 'initial')
    contrast_choice_single(bilayer, contrast_range, [6.36], angle_times, save_path, 'D2O')
    contrast_choice_single(bilayer, contrast_range, [-0.56], angle_times, save_path, 'H2O')
    contrast_choice_single(bilayer, contrast_range, [-0.56, 6.36], angle_times, save_path, 'H2O_D2O')

    contrast_range = np.linspace(-0.55, 6.36, 50)
    contrast_choice_double(bilayer, contrast_range, angle_times, save_path)

    bilayer.nested_sampling([6.36, 6.36], angle_times, save_path, 'D2O_D2O', dynamic=True)
    bilayer.nested_sampling([-0.56, 6.36], angle_times, save_path, 'H2O_D2O', dynamic=True)

def underlayer_results(save_path):
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    bilayer = SymmetricBilayer()
    contrasts = [6.36]
    angle_times = [(0.5, 70, 10), (2.3, 70, 40)]

    thickness_range = np.linspace(5, 500, 50)
    sld_range = np.linspace(1, 9, 100)
    thick, sld = underlayer_choice(bilayer, thickness_range, sld_range, contrasts, angle_times, save_path)
    print('Thickness: {}'.format(round(thick)))
    print('SLD: {}'.format(round(sld, 2)))

if __name__ == '__main__':
    save_path = './results'

    angle_results_normal(save_path)
    angle_results_bilayer(save_path)
    contrast_results(save_path)
    underlayer_results(save_path)
