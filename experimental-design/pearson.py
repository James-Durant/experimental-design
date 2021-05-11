import matplotlib.pyplot as plt
import numpy as np
import os

from numpy.typing import ArrayLike

from simulate import AngleTimes
from simulate import simulate_single_contrast as simulate

from utils import save_plot
from utils import fisher_multiple_contrasts as fisher

from structures import Bilayer

def pearson(bilayer: Bilayer, initial_contrasts: ArrayLike, contrasts: ArrayLike,
            angle_times: AngleTimes, save_path: str) -> None:
    xi = bilayer.parameters

    models_init, qs_init, counts_init = [], [], []
    for contrast in initial_contrasts:
        sample = bilayer.using_contrast(contrast)
        simulated = simulate(sample, angle_times, include_counts=True)
        models_init.append(simulated[0])
        qs_init.append(simulated[1].x)
        counts_init.append(simulated[2])
        
    pearson_rs = {param: [] for param in xi}
    for x, new_contrast in enumerate(contrasts):
        sample = bilayer.using_contrast(new_contrast)
        model_new, data_new, counts_new = simulate(sample, angle_times, include_counts=True)

        qs, counts, models = qs_init.copy(), counts_init.copy(), models_init.copy()
        qs.append(data_new.x)
        counts.append(counts_new)
        models.append(model_new)
        
        g = fisher(qs, xi, counts, models)
        f = np.linalg.inv(g)

        for i, param in enumerate(xi):
            rs = [f[i,j] / (np.sqrt(f[i,i]) * np.sqrt(f[j,j])) for j in range(len(xi))]
            pearson_rs[param].append(rs)

        if x % 30 == 0:
            print('>>> {0}/{1}'.format(x, len(contrasts)))

    labels = [param.name for param in xi]
    save_path = os.path.join(save_path, str(bilayer))

    fig = plt.figure(figsize=[36,14])

    # Iterate over each parameter.
    for i, param in enumerate(xi):
        ax = fig.add_subplot(2, 4, i+1)

        # Iterate over all other parameters.
        rs = np.array(pearson_rs[param])
        for j in range(len(xi)):
            ax.plot(contrasts, rs[:,j], label=labels[j])

        # Set the plot title to the first parameter in each parameter pair.
        ax.set_title(param.name)
        ax.legend(loc='upper right')
        ax.set_xlabel("$\mathregular{Contrast\ SLD\ (10^{-6} \AA^{-2})}$", fontsize=11, weight='bold')
        ax.set_ylabel("Pearson Correlation", fontsize=11, weight='bold')
        ax.set_ylim(-1.1, 1.1)

    save_path = os.path.join(save_path, str(bilayer))
    save_plot(fig, save_path, 'pearson_contrast')

if __name__ == '__main__':
    from structures import SymmetricBilayer
    from structures import SingleAsymmetricBilayer, DoubleAsymmetricBilayer

    save_path = './results'

    bilayer = SymmetricBilayer()
    initial_contrasts = [6.36]
    contrasts = np.arange(-0.55, 6.36, 0.05)
    angle_times = {0.7: (70, 10),
                   2.0: (70, 40)}
    
    pearson(bilayer, initial_contrasts, contrasts, angle_times, save_path)