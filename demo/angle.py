import numpy as np
import sys
# Add to system path to access experimental design code.
sys.path.append('../experimental-design')

from refnx.reflect import SLD

from structures import Sample
from design import angle_choice_single

# Defines a structure describing a simple sample.
air = SLD(0, name='Air')
layer1 = SLD(4, name='Layer 1')(thick=200, rough=2)
layer2 = SLD(6, name='Layer 2')(thick=6, rough=2)
substrate = SLD(2.047, name='Substrate')(thick=0, rough=2)

structure = air | layer1 | layer2 | substrate
structure.name = 'thin_layer_sample'

# Wrap the refnx structure in the custom Sample class.
sample = Sample(structure)

# Path to directory to save results to.
save_path = './results'

points = 70 # Number of data points to simulate.
time = 40 # Time to use for simulation.

# Angles (in degrees) to calculate over.
angle_range = np.linspace(0.2, 2.3, 200)

# Investigate first angle choice.
initial_angle_times = {}
angle_choice_single(sample, initial_angle_times, angle_range, points, time, save_path, 'angle_choice_1')

# Investigate second angle choice.
initial_angle_times = {2.3: (70, 40)} # First first angle choice - Angle: (Points, Time)
angle_choice_single(sample, initial_angle_times, angle_range, points, time, save_path, 'angle_choice_2')
