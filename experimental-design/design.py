import matplotlib.pyplot as plt
import numpy as np
import os

from itertools import combinations
from utils import fisher, save_plot

from structures import VariableAngle, VariableContrast, VariableUnderlayer

def angle_choice_single(sample, initial_angle_times, angle_range, points_new, time_new, save_path, filename, contrasts=[]):
    assert isinstance(sample, VariableAngle)

    xi = sample.parameters
    qs_init, counts_init, models_init = sample.angle_info(initial_angle_times, contrasts)

    min_eigs = []
    for i, angle_new in enumerate(angle_range):
        new_angle_times = {angle_new: (points_new, time_new)}

        qs_new, counts_new, models_new = sample.angle_info(new_angle_times, contrasts)
        qs = qs_init + qs_new
        counts = counts_init + counts_new
        models = models_init + models_new

        g = fisher(qs, xi, counts, models)
        min_eigs.append(np.linalg.eigvalsh(g)[0])

        if i % 100 == 0:
            print('>>> {0}/{1}'.format(i, len(angle_range)))

    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    ax.plot(angle_range, min_eigs)

    ax.set_xlabel('Angle (°)', fontsize=11, weight='bold')
    ax.set_ylabel('Minimum Eigenvalue (arb.)', fontsize=11, weight='bold')

    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, filename)

    return angle_range[np.argmax(min_eigs)]

def angle_choice_double(sample, angle_range, points_new, time_new, save_path, contrasts=[]):
    assert isinstance(sample, VariableAngle)

    xi = sample.parameters
    angles = np.asarray(list(combinations(angle_range, 2)))

    min_eigs = []
    for i, (angle_1, angle_2) in enumerate(angles):
        angle_times = {angle_1: (points_new, time_new),
                       angle_2: (points_new, time_new)}

        qs, counts, models = sample.angle_info(angle_times, contrasts)

        g = fisher(qs, xi, counts, models)
        min_eigs.append(np.linalg.eigvalsh(g)[0])

        # Display progress.
        if i % 500 == 0:
            print('>>> {0}/{1}'.format(i, len(angles)))

    x = np.concatenate([angles[:,0], angles[:,1]])
    y = np.concatenate([angles[:,1], angles[:,0]])

    min_eigs.extend(min_eigs)

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, min_eigs, cmap='Spectral')

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel("$\mathregular{Angle \ 1 \ (°)}$", fontsize=11, weight='bold')
    ax.set_ylabel("$\mathregular{Angle \ 2 \ (°)}$", fontsize=11, weight='bold')
    ax.set_zlabel('Minimum Eigenvalue', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'angle_choice_double')

    maximum = np.argmax(min_eigs)
    return x[maximum], y[maximum]

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

    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    ax.plot(contrast_range, min_eigs)

    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel("Minimum Eigenvalue", fontsize=11, weight='bold')

    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'contrast_choice_single_'+filename)

    return contrast_range[np.argmax(min_eigs)]

def contrast_choice_double(sample, contrast_range, angle_times, save_path):
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

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel("$\mathregular{Contrast \ 1 \ SLD \ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel("$\mathregular{Contrast \ 2 \ SLD \ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_zlabel('Minimum Eigenvalue', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'contrast_choice_double')

    maximum = np.argmax(min_eigs)
    return x[maximum], y[maximum]

def underlayer_choice(sample, thickness_range, sld_range, contrasts, angle_times, save_path):
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

    fig = plt.figure(figsize=[9,7])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, min_eigs, cmap='Spectral')

    ax.set_ylabel('$\mathregular{Underlayer\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_xlabel('$\mathregular{Underlayer\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_zlabel('Minimum Eigenvalue', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'underlayer_choice')

    maximum = np.argmax(min_eigs)
    return x[maximum], y[maximum]

def angle_results_normal(save_path):
    from structures import STRUCTURES

    points = 70
    time = 40

    angle_range = np.linspace(0.2, 2.4, 500)
    for structure in STRUCTURES:
        angle_times = {}
        for i in range(4):
            filename = 'angle_choice_single_{}'.format(i+1)
            angle = angle_choice_single(structure(), angle_times, angle_range, points, time, save_path, filename)
            if angle in angle_times:
                angle_times[angle] = (angle_times[angle][0]+points, angle_times[angle][1]+time)
            else:
                angle_times[angle] = (points, time)

    angle_range = np.linspace(0.2, 2.4, 50)
    for structure in STRUCTURES:
        angle_choice_double(structure(), angle_range, points, time, save_path)

def angle_results_bilayer(save_path):
    from structures import BILAYERS

    points = 70
    time = 20
    angle_range = np.linspace(0.2, 2.4, 500)
    contrasts = [-0.56, 6.36]

    for bilayer in BILAYERS:
        angle_times = {}
        for i in range(4):
            filename = 'angle_choice_single_{}'.format(i+1)
            angle = angle_choice_single(bilayer(), angle_times, angle_range, points, time, save_path, filename, contrasts)
            if angle in angle_times:
                angle_times[angle] = (angle_times[angle][0]+points, angle_times[angle][1]+time)
            else:
                angle_times[angle] = (points, time)

    angle_range = np.linspace(0.2, 2.4, 50)
    for structure in BILAYERS:
        angle_choice_double(structure(), angle_range, points, time, save_path, contrasts)

def contrast_results(save_path):
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    bilayer = SymmetricBilayer()
    angle_times = {0.7: (70, 10),
                   2.3: (70, 40)}

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
    angle_times = {0.7: (70, 10),
                   2.3: (70, 40)}

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
