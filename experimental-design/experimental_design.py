import matplotlib.pyplot as plt
import numpy as np
import os

from simulate import simulate
from utils import fisher, vary_structure, save_plot

def angle_choice(sample, initial_angle_times, angles, points_new, time_new, save_path):
    xi = vary_structure(sample)

    qs_init, counts_init, models_init = [], [], []
    if initial_angle_times:
        model, data = simulate(sample, initial_angle_times)
        
        qs_init.append(data[:,0])
        counts_init.append(data[:,3])
        models_init.append(model)
        
    metric = []
    for i, angle_new in enumerate(angles):        
        angle_times = {angle_new: (points_new, time_new)}
        model, data = simulate(sample, angle_times)

        qs, counts, models = qs_init.copy(), counts_init.copy(), models_init.copy()
        qs.append(data[:,0])
        counts.append(data[:,3])
        models.append(model)
        
        g = fisher(qs, xi, counts, models)
        f = np.linalg.inv(g)
        metric.append(np.sum(f))

        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(angles)))
    
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    ax.plot(angles, metric)

    ax.set_xlabel('Angle (°)', fontsize=11, weight='bold')
    ax.set_ylabel('Metric (arb.)', fontsize=11, weight='bold')
    
    if not initial_angle_times:
        ax.set_yscale('log')
    
    filename = 'angle_choice' if initial_angle_times else 'angle_choice_initial'
    for angle in initial_angle_times:
        filename = filename + '_' + str(angle).replace('.', '')
        
    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, filename)

def contrast_choice(bilayer, initial_contrasts, contrasts, angle_times, save_path, filename):
    xi = bilayer.parameters

    qs_init, counts_init, models_init = [], [], []
    for contrast in initial_contrasts:
        model, data = simulate(bilayer.using_contrast(contrast), angle_times)
        
        qs_init.append(data[:,0])
        counts_init.append(data[:,3])
        models_init.append(model)
        
    valid_contrasts, metric = [], []
    for i, new_contrast in enumerate(contrasts):
        model, data = simulate(bilayer.using_contrast(new_contrast), angle_times)

        qs, counts, models = qs_init.copy(), counts_init.copy(), models_init.copy()
        qs.append(data[:,0])
        counts.append(data[:,3])
        models.append(model)
        
        g = fisher(qs, xi, counts, models)
        try:
            f = np.linalg.inv(g)
            metric.append(np.sum(abs(f)))
            valid_contrasts.append(new_contrast)
        except np.linalg.LinAlgError:
            continue

        # Display progress.
        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(contrasts)))
    
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    ax.plot(valid_contrasts, metric)

    ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
    ax.set_ylabel("Metric", fontsize=11, weight='bold')
    
    if len(initial_contrasts) < 2:
        ax.set_yscale('log')
    
    save_path = os.path.join(save_path, bilayer.name)
    save_plot(fig, save_path, 'contrast_choice_'+filename)    

def underlayer_choice(bilayer, thicknesses, slds, contrast_slds, angle_times, save_path):
    xi = [param for param in bilayer.parameters if param.name != 'SiO2 Thickness']

    x, y, metric = [], [], []
    for i, thickness in enumerate(thicknesses):
        for sld in slds:
            bilayer.sio2_thick = thickness
            bilayer.sio2_sld = sld
            x.append(thickness)
            y.append(sld)
            
            qs, counts, models = [], [], []
            for contrast_sld in contrast_slds:
                model, data = simulate(bilayer.using_contrast(contrast_sld), angle_times)
                
                qs.append(data[:,0])
                counts.append(data[:,3])
                models.append(model)
    
            g = fisher(qs, xi, counts, models)
            f = np.linalg.inv(g)

            metric.append(np.sum(np.abs(f)))

        if i % 5 == 0:
            print('>>> {0}/{1}'.format(i, len(thicknesses)))

    fig = plt.figure(figsize=[9,7])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, np.log(metric), cmap=plt.cm.Spectral)

    ax.set_ylim(ax.get_ylim()[::-1])        
    ax.set_xlabel('$\mathregular{Underlayer\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_ylabel('$\mathregular{Underlayer\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_zlabel('Log Metric (arb.)', fontsize=11, weight='bold')

    save_path = os.path.join(save_path, bilayer.name)
    save_plot(fig, save_path, 'underlayer_choice')

    minimum = np.argmin(metric)
    return x[minimum], y[minimum]

def angle_results():
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import simple_sample, QCS_sample, many_param_sample
    
    save_path = './results'
    
    structure = simple_sample()
    points = 70
    time = 5
    angles = np.arange(0.2, 2.4, 0.01)
    
    for angle_times in [{},
                        {0.7: (points, time)},
                        {0.7: (points, time), 2.0: (points, 4*time)}]:
        angle_choice(structure, angle_times, angles, points, time, save_path)

def contrast_results():
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    save_path = './results'

    bilayer = SymmetricBilayer()
    contrasts = np.arange(-0.55, 6.36, 0.05)
    angle_times = {0.7: (70, 10),
                   2.0: (70, 40)}
    
    contrast_choice(bilayer, [], contrasts, angle_times, save_path, 'initial')
    contrast_choice(bilayer, [6.36], contrasts, angle_times, save_path, 'D2O')
    contrast_choice(bilayer, [-0.56], contrasts, angle_times, save_path, 'H2O')
    contrast_choice(bilayer, [-0.56, 6.36], contrasts, angle_times, save_path, 'H2O_D2O')
    
def underlayer_results():
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer
    
    save_path = './results'

    bilayer = SymmetricBilayer()
    contrasts = np.arange(-0.55, 6.36, 0.05)
    angle_times = {0.7: (70, 10),
                   2.0: (70, 40)}
    
    thicknesses = np.arange(5, 500, 1)
    slds = np.arange(1, 9, 0.1)
    thick, sld = underlayer_choice(bilayer, thicknesses, slds, contrasts, angle_times, save_path)
    print('Thickness: {}'.format(thick))
    print('SLD: {}'.format(sld))

    bilayer.parameters = [param for param in bilayer.parameters if param.name != 'SiO2 Thickness']

    bilayer.sample([6.36], angle_times, save_path, 'normal')
    
    bilayer.sio2_sld = 8.9
    bilayer.sio2_thick = 80
    bilayer.sample([6.36], angle_times, save_path, 'new')

if __name__ == '__main__':
    angle_results()
    contrast_results()
    underlayer_results()
