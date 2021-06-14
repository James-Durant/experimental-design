import matplotlib.pyplot as plt
import numpy as np
import os
import refnx.reflect

from itertools import combinations

from simulate import simulate
from utils import fisher, vary_structure, save_plot

from structures import Bilayer

def angle_choice_single(sample, initial_angle_times, angle_range, points_new, time_new, save_path, contrasts=[]):
    if isinstance(sample, refnx.reflect.Structure):
        xi = vary_structure(sample)
        qs_init, counts_init, models_init = [], [], []
        if initial_angle_times:
            model, data = simulate(sample, initial_angle_times)
        
            qs_init.append(data[:,0])
            counts_init.append(data[:,3])
            models_init.append(model)
     
    elif isinstance(sample, Bilayer):
        xi = sample.parameters
        qs_init, counts_init, models_init = sample.contrast_information(initial_angle_times, contrasts)
              
    else:
        raise RuntimeError('invalid sample given')
        
    min_eigs = []
    for i, angle_new in enumerate(angle_range):    
        angle_times = {angle_new: (points_new, time_new)}
        
        if isinstance(sample, refnx.reflect.Structure):
            model_new, data_new = simulate(sample, angle_times)

            qs = qs_init + [data_new[:,0]]
            counts = counts_init + [data_new[:,3]]
            models = models_init + [model_new]
        
        elif isinstance(sample, Bilayer):
            qs_new, counts_new, models_new = sample.contrast_information(angle_times, contrasts)
            qs = qs_init + qs_new
            counts = counts_init + counts_new
            models = models_init + models_new
        
        g = fisher(qs, xi, counts, models)
        min_eigs.append(np.linalg.eigvalsh(g)[0])

        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(angle_range)))
    
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    ax.plot(angle_range, min_eigs)

    ax.set_xlabel('Angle (°)', fontsize=11, weight='bold')
    ax.set_ylabel('Minimum Eigenvalue (arb.)', fontsize=11, weight='bold')
    
    filename = 'angle_choice_single' if initial_angle_times else 'angle_choice_single_initial'
    for angle in initial_angle_times:
        filename = filename + '_' + str(angle).replace('.', '')
        
    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, filename)

    return angle_range[np.argmax(min_eigs)]
def contrast_choice_single(bilayer, contrast_range, contrasts, angle_times, save_path, filename):
    xi = bilayer.parameters
    qs_init, counts_init, models_init = [], [], []
    for contrast in initial_contrasts:
        model, data = simulate(bilayer.using_contrast(contrast), angle_times)
        
        qs_init.append(data[:,0])
        counts_init.append(data[:,3])
        models_init.append(model)
    
    lb = np.array([param.bounds.lb for param in xi])
    ub = np.array([param.bounds.ub for param in xi])
    J = np.diag(1/(ub-lb))
    
    metrics = []
    for i, new_contrast in enumerate(contrasts):
        model, data = simulate(bilayer.using_contrast(new_contrast), angle_times)
    xi = bilayer.parameters
    
    qs_init, counts_init, models_init = bilayer.contrast_information(angle_times, contrasts)
    
    min_eigs = []
    for contrast in contrast_range:
        qs_new, counts_new, models_new = bilayer.contrast_information(angle_times, [contrast])
        
        qs = qs_init + qs_new
        counts = counts_init + counts_new
        models = models_init + models_new
        
        g = fisher(qs, xi, counts, models)
        min_eigs.append(np.linalg.eigvalsh(g)[0])

        # Display progress.
        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(contrast_range)))
    
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    ax.plot(contrast_range, min_eigs)
    
    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel("Minimum Eigenvalue", fontsize=11, weight='bold')
    
    #if len(initial_contrasts) < 2:
    #    ax.set_yscale('log')
    
    save_path = os.path.join(save_path, bilayer.name)
    save_plot(fig, save_path, 'contrast_choice_single_'+filename)    

def contrast_choice_double(bilayer, contrast_range, angle_times, save_path):
    xi = bilayer.parameters
    contrasts = np.asarray(list(combinations(contrast_range, 2)))

    metrics = []
    for i, (contrast_1, contrast_2) in enumerate(contrasts):
        model_1, data_1 = simulate(bilayer.using_contrast(contrast_1), angle_times)
        model_2, data_2 = simulate(bilayer.using_contrast(contrast_2), angle_times)

        qs = [data_1[:,0], data_2[:,0]]
        counts = [data_1[:,3], data_2[:,3]]
        models = [model_1, model_2]
        
        g = fisher(qs, xi, counts, models)
        metrics.append(np.linalg.eigvalsh(g)[0])

        # Display progress.
        if i % 500 == 0:
            print('>>> {0}/{1}'.format(i, len(contrasts)))

    x = np.concatenate([contrasts[:,0], contrasts[:,1]])
    y = np.concatenate([contrasts[:,1], contrasts[:,0]])
    
    metrics.extend(metrics)

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, metrics, cmap='Spectral')
    
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlabel("$\mathregular{Contrast \ 1 \ SLD \ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel("$\mathregular{Contrast \ 2 \ SLD \ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_zlabel('Minimum Eigenvalue', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    save_path = os.path.join(save_path, bilayer.name)
    save_plot(fig, save_path, 'contrast_choice_double')

    maximum = np.argmax(metrics)
    return x[maximum], y[maximum]

def underlayer_choice(bilayer, thickness_range, sld_range, contrasts, angle_times, save_path):
    xi = bilayer.parameters
    
    x, y, min_eigs = [], [], []
    for i, thickness in enumerate(thickness_range):
        for sld in sld_range:
            qs, counts, models = bilayer.contrast_information(angle_times, contrasts,
                                                              underlayer=(sld, thickness))
            x.append(thickness)
            y.append(sld)
            
            g = fisher(qs, xi, counts, models)
            min_eigs.append(np.linalg.eigvalsh(g)[0])
            
        if i % 5 == 0:
            print('>>> {0}/{1}'.format(i, len(thickness_range)))

    fig = plt.figure(figsize=[9,7])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(y, x, min_eigs, cmap='Spectral')
    
    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_ylabel('$\mathregular{Underlayer\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_xlabel('$\mathregular{Underlayer\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_zlabel('Minimum Eigenvalue', fontsize=11, weight='bold')

    save_path = os.path.join(save_path, bilayer.name)
    save_plot(fig, save_path, 'underlayer_choice')

    maximum = np.argmax(min_eigs)
    return x[maximum], y[maximum]

def angle_results_normal(save_path='./results'):
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import simple_sample, QCS_sample, many_param_sample
    
    structure = many_param_sample()
    
    points = 70
    time = 5
    angle_range = np.linspace(0.2, 2.4, 500)
    
    initial = {}
    first = {0.7: (points, time)}
    second = {0.7: (points, time), 2.0: (points, time)}
    
    for angle_times in [initial, first, second]:
        angle_choice_single(structure, angle_times, angle_range, points, time, save_path)

def angle_results_bilayer(save_path='./results'):
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer
    
    bilayer = SymmetricBilayer()
    
    points = 70
    time = 10
    angle_range = np.linspace(0.2, 2.4, 500)
    contrasts = [-0.56, 6.36]
    
    initial = {}
    first = {0.7: (points, time)}
    second = {0.7: (points, time), 2.0: (points, time)}
    
    for angle_times in [initial, first, second]:
        angle_choice_single(bilayer, angle_times, angle_range, points, time, save_path)

def contrast_results():
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    save_path = './results'

    bilayer = SymmetricBilayer()
    contrast_range = np.linspace(-0.55, 6.36, 60)
    angle_times = {0.7: (70, 10),
                   2.3: (70, 40)}
    
    contrast_choice_single(bilayer, contrast_range, [], angle_times, save_path, 'initial')
    contrast_choice_single(bilayer, contrast_range, [6.36], angle_times, save_path, 'D2O')
    contrast_choice_single(bilayer, contrast_range, [-0.56], angle_times, save_path, 'H2O')
    contrast_choice_single(bilayer, contrast_range, [-0.56, 6.36], angle_times, save_path, 'H2O_D2O')
    
    contrast_choice_double(bilayer, contrast_range, angle_times, save_path)
    
def underlayer_results():
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer
    
    save_path = './results'

    bilayer = SymmetricBilayer()
    contrasts = [6.36]
    angle_times = {0.7: (70, 10),
                   2.3: (70, 40)}
    
    thickness_range = np.linspace(5, 500, 150)
    sld_range = np.linspace(1, 9, 150)
    thick, sld = underlayer_choice(bilayer, thickness_range, sld_range, contrasts, angle_times, save_path)
    print('Thickness: {}'.format(thick))
    print('SLD: {}'.format(sld))

if __name__ == '__main__':
    #angle_results_normal()
    #angle_results_bilayer()
    contrast_results()
    #underlayer_results()
