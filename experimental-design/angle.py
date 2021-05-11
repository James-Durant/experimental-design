import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Callable
from numpy.typing import ArrayLike

from simulate import AngleTimes
from simulate import simulate_single_contrast as simulate

from utils import vary_structure, save_plot
from utils import fisher_multiple_contrasts as fisher

def angle_choice(structure: Callable, initial_angle_times: AngleTimes,
                 angles: ArrayLike, points_new: int, time_new: float, save_path: str) -> None:
    sample, xi = vary_structure(structure())

    models_init, qs_init, counts_init = [], [], []
    if initial_angle_times:
        simulated = simulate(sample, initial_angle_times, include_counts=True)
        
        models_init.append(simulated[0])
        qs_init.append(simulated[1].x)
        counts_init.append(simulated[2])
        
    metric = []
    for i, angle_new in enumerate(angles):        
        angle_times = {angle_new: (points_new, time_new)}
        model_new, data_new, counts_new = simulate(sample, angle_times, include_counts=True)

        qs, counts, models = qs_init.copy(), counts_init.copy(), models_init.copy()
        qs.append(data_new.x)
        counts.append(counts_new)
        models.append(model_new)
        
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
        
    save_path = os.path.join(save_path, structure.__name__)
    save_plot(fig, save_path, filename)
    
if __name__ == '__main__':
    from structures import similar_sld_sample_1, similar_sld_sample_2
    from structures import thin_layer_sample_1, thin_layer_sample_2
    from structures import easy_sample, QCS_sample, many_param_sample
    
    save_path = './results'
    
    structure = easy_sample
    points = 70
    time = 5
    angles = np.arange(0.2, 2.4, 0.01)
    
    for angle_times in [{},
                        {0.7: (points, time)},
                        {0.7: (points, time), 2.0: (points, 4*time)}]:
        angle_choice(structure, angle_times, angles, points, time, save_path)
