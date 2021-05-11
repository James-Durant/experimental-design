import matplotlib.pyplot as plt
import numpy as np
import os

from typing import List
from numpy.typing import ArrayLike

from simulate import AngleTimes
from simulate import simulate_single_contrast as simulate

from utils import save_plot
from utils import fisher_multiple_contrasts as fisher

from structures import Bilayer

def contrast_choice(bilayer: Bilayer, initial_contrasts: List[float], contrasts: ArrayLike,
                    angle_times: AngleTimes, save_path: str, filename: str) -> None:
    xi = bilayer.parameters

    models_init, qs_init, counts_init = [], [], []
    for contrast in initial_contrasts:
        sample = bilayer.using_contrast(contrast)
        simulated = simulate(sample, angle_times, include_counts=True)
        models_init.append(simulated[0])
        qs_init.append(simulated[1].x)
        counts_init.append(simulated[2])
        
    metric = []
    valid_contrasts = []
    for i, new_contrast in enumerate(contrasts):
        sample = bilayer.using_contrast(new_contrast)
        model_new, data_new, counts_new = simulate(sample, angle_times, include_counts=True)

        qs, counts, models = qs_init.copy(), counts_init.copy(), models_init.copy()
        qs.append(data_new.x)
        counts.append(counts_new)
        models.append(model_new)
        
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
    
    save_path = os.path.join(save_path, str(bilayer))
    save_plot(fig, save_path, 'contrast_choice_'+filename)
    
if __name__ == '__main__':
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
