import numpy as np
import sys
# Add to system path to access experimental design code.
sys.path.append('../experimental-design')

from refnx.reflect import SLD
from experimental_design import angle_choice

def simple_sample():
    """Defines a structure describing a simple sample."""
    air = SLD(0, name='Air')
    layer1 = SLD(4, name='Layer 1')(thick=100, rough=2)
    layer2 = SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)
    return air | layer1 | layer2 | substrate

# Path to directory to save results to.
save_path = './results'

points = 70 # Number of data points to simulate.
time = 20 # Time to use for simulation.

# Angles (in degrees) to calculate over.
angles = np.arange(0.25, 2.5, 0.01)

# Investigate first angle choice.
initial_angle_times = {}
angle_choice(simple_sample, initial_angle_times, angles, points, time, save_path)

# Investigate second angle choice.
initial_angle_times = {0.7: (70, 5)} # First first angle choice - Angle: (Points, Time)
angle_choice(simple_sample, initial_angle_times, angles, points, time, save_path)
