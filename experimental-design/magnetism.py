import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
plt.rcParams['figure.dpi'] = 600

from magnetic import SampleYIG
from utils import save_plot

def magnetism(yig_thick_range, pt_thick_range, angle_times, save_path, save_views=False):
    sample = SampleYIG()
    sample.Pt_mag.value = 0.01638
    
    x, y, infos = [], [], []
    for i, yig_thick in enumerate(yig_thick_range):
        # Display progress.
        if i % 5 == 0:
            print('>>> {0}/{1}'.format(i*len(pt_thick_range), len(pt_thick_range)*len(yig_thick_range)))
        
        for pt_thick in pt_thick_range:
            g = sample.underlayer_info(angle_times, yig_thick, pt_thick)
            
            infos.append(g[0,0])              
            x.append(yig_thick)
            y.append(pt_thick)

    fig = plt.figure(figsize=[10,8])
    ax = fig.add_subplot(111, projection='3d')
    
    surface = ax.plot_trisurf(x, y, infos, cmap='plasma')
    fig.colorbar(surface, fraction=0.046, pad=0.04)

    ax.set_xlabel('$\mathregular{YIG\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_ylabel('$\mathregular{Pt\ Thickness\ (\AA)}$', fontsize=11, weight='bold')
    ax.set_zlabel('Fisher Information', fontsize=11, weight='bold')
    ax.ticklabel_format(axis='z', style='sci', scilimits=(0,0))

    # Save the plot.
    save_path = os.path.join(save_path, sample.name)
    save_plot(fig, save_path, 'underlayer_choice')

    if save_views:
        save_path = os.path.join(save_path, 'underlayer_choice')
        for i in range(0, 360, 10):
            ax.view_init(elev=40, azim=i)
            save_plot(fig, save_path, 'underlayer_choice_{}'.format(i))
            
    maximum = np.argmax(infos)
    return x[maximum], y[maximum]
            
def _magnetism_results(save_path='./results'):
    yig_thick_range = np.linspace(400, 900, 75)
    pt_thick_range = np.linspace(20, 100, 50)
    
    angle_times = [(0.5, 100, 20),
                   (1.0, 100, 40),
                   (2.0, 100, 80)]
    
    yig_thick, pt_thick = magnetism(yig_thick_range, pt_thick_range, angle_times, save_path, save_views=True)
    print('YIG Thickness: {}'.format(round(yig_thick, 1)))
    print('Pt Thickness: {}'.format(round(pt_thick, 1)))
    
if __name__ == '__main__':
    _magnetism_results()
