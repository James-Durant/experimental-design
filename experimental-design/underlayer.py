import matplotlib.pyplot as plt

import numpy as np
import os

from simulate import simulate_single_contrast as simulate

from utils import save_plot
from utils import fisher_multiple_contrasts as fisher

def underlayer_choice(bilayer, thicknesses, slds, contrast_slds, angle_times, save_path):
    xi = [param for param in bilayer.parameters if param.name != 'SiO2 Thickness']

    x, y, metric = [], [], []
    for i, thickness in enumerate(thicknesses):
        for sld in slds:
            bilayer.sio2_thick = thickness
            bilayer.sio2_sld = sld
            x.append(thickness)
            y.append(sld)
            
            models, datasets, counts = [], [], []
            for contrast_sld in contrast_slds:
                structure = bilayer.using_contrast(contrast_sld)
                simulated = simulate(structure, angle_times, include_counts=True)
                models.append(simulated[0])
                datasets.append(simulated[1].x)
                counts.append(simulated[2])
    
            g = fisher(datasets, xi, counts, models)
            f = np.linalg.inv(g)

            metric.append(np.sum(np.abs(f)))

        if i % 50 == 0:
            print('>>> {0}/{1}'.format(i, len(thicknesses)))

    fig = plt.figure(figsize=[9,7])
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x, y, np.log(metric), cmap=plt.cm.Spectral)

    ax.set_ylim(ax.get_ylim()[::-1])        
    ax.set_xlabel('$\mathregular{Underlayer\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_ylabel('$\mathregular{Underlayer\ SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
    ax.set_zlabel('Log Metric (arb.)', fontsize=11, weight='bold')

    save_path = os.path.join(save_path, str(bilayer))
    save_plot(fig, save_path, 'underlayer_choice')

    minimum = np.argmin(metric)
    return x[minimum], y[minimum]

if __name__ == '__main__':
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer
    
    save_path = './results'
    
    bilayer = SymmetricBilayer()
    contrasts = [6.36]
    angle_times = {0.7: (70, 10),
                   2.0: (70, 40)}
    
    thicknesses = np.arange(0, 500, 1)
    slds = np.arange(1, 9, 0.1)
    thick, sld = underlayer_choice(bilayer, thicknesses, slds, contrasts, angle_times, save_path)
    print('Thickness: {}'.format(thick))
    print('SLD: {}'.format(sld))

    #bilayer.parameters = [param for param in bilayer.parameters if param.name != 'SiO2 Thickness']

    #bilayer.sample([6.36], angle_times, './results', 'normal')
    
    #bilayer.sio2_sld = 8.9
    #bilayer.sio2_thick = 61
    #bilayer.sample([6.36], angle_times, './results', 'new')