import numpy as np
import sys
# Add to system path to access experimental design code.
sys.path.append('../experimental-design')

from refnx.reflect import SLD, Slab
from refnx.analysis import Parameter

from experimental_design import contrast_choice

class Bilayer():
    """Defines a model describing an asymmetric bilayer."""

    def __init__(self):
        self.name = 'asymmetric_bilayer'
        
        # Define known values.
        self.si_sld    =  2.07
        self.sio2_sld  =  3.41
        self.pc_hg_sld =  1.98
        self.dPC_tg    =  7.45
        self.hLPS_tg   = -0.37
        self.core_D2O  =  4.2
        self.core_H2O  =  2.01

        # Define the parameters of the model.
        self.si_rough       = Parameter(5.5,    'Si/SiO2 Roughness',         (3,8))
        self.sio2_thick     = Parameter(13.4,   'SiO2 Thickness',            (10,30))
        self.sio2_rough     = Parameter(3.2,    'SiO2/Bilayer Roughness',    (2,5))
        self.sio2_solv      = Parameter(0.038,  'SiO2 Hydration',            (0,0.5))
        self.inner_hg_thick = Parameter(9.0,    'Inner Headgroup Thickness', (5,20))
        self.inner_hg_solv  = Parameter(0.39,   'Inner Headgroup Hydration', (0,1))
        self.bilayer_rough  = Parameter(4.0,    'Bilayer Roughness',         (0,12))
        self.inner_tg_thick = Parameter(16.7,   'Inner Tailgroup Thickness', (10,20))
        self.outer_tg_thick = Parameter(14.9,   'Outer Tailgroup Thickness', (10,20))
        self.tg_solv        = Parameter(0.0085, 'Tailgroup Hydration',       (0,1))
        self.core_thick     = Parameter(28.7,   'Core Thickness',            (0,50))
        self.core_solv      = Parameter(0.26,   'Core Hydration',            (0,1))
        self.asym_value     = Parameter(0.95,   'Asymmetry Value',           (0,1))

        self.inner_tg_sld = SLD(self.asym_value*self.dPC_tg  + (1-self.asym_value)*self.hLPS_tg)
        self.outer_tg_sld = SLD(self.asym_value*self.hLPS_tg + (1-self.asym_value)*self.dPC_tg)

        self.parameters = [self.si_rough,
                           self.sio2_thick,
                           self.sio2_rough,
                           self.sio2_solv,
                           self.inner_hg_thick,
                           self.inner_hg_solv,
                           self.bilayer_rough,
                           self.inner_tg_thick,
                           self.outer_tg_thick,
                           self.tg_solv,
                           self.core_thick,
                           self.core_solv,
                           self.asym_value]

        # Vary all of the parameters defined above.
        for param in self.parameters:
            param.vary=True

    def using_contrast(self, contrast_sld):
        """Creates a structure representing the bilayer measured using a
           contrast of given `contrast_sld`.

        Args:
            contrast_sld (float): SLD of contrast to simulate.

        Returns:
            refnx.reflect.Structure: structure in given contrast.

        """
        # Rescales the H2O to D2O scale from 0 to 1.
        contrast_point = (contrast_sld + 0.56) / (6.35 + 0.56)
        # Calculate core SLD with the given contrast SLD.
        core_sld = contrast_point*self.core_D2O + (1-contrast_point)*self.core_H2O

        substrate = SLD(self.si_sld)
        solution  = SLD(contrast_sld)(0, self.bilayer_rough)

        sio2     = Slab(self.sio2_thick,     self.sio2_sld,     self.si_rough,      vfsolv=self.sio2_solv)
        inner_hg = Slab(self.inner_hg_thick, self.pc_hg_sld,    self.sio2_rough,    vfsolv=self.inner_hg_solv)
        inner_tg = Slab(self.inner_tg_thick, self.inner_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        outer_tg = Slab(self.outer_tg_thick, self.outer_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        core     = Slab(self.core_thick,     core_sld,          self.bilayer_rough, vfsolv=self.core_solv)

        return substrate | sio2 | inner_hg | inner_tg | outer_tg | core | solution

# Path to directory to save results to.
save_path = './results'

# Number of points and times to simulate for each angle.
angle_times = {0.7: (70, 100),
               2.0: (70, 400)} # Angle: (Points, Time)

# Contrast SLDs to calculate over.
contrasts = np.arange(-0.55, 6.36, 0.05)

# Initial contrast choice.
contrast_choice(Bilayer(), [], contrasts, angle_times, save_path, 'initial')

# Second contrast choice, assuming D2O was first measured.
contrast_choice(Bilayer(), [6.36], contrasts, angle_times, save_path, 'D2O')

# Second contrast choice, assuming H2O was first measured.
contrast_choice(Bilayer(), [-0.56], contrasts, angle_times, save_path, 'H2O')

# Third contrast choice, assuming D2O and H2O were measured.
contrast_choice(Bilayer(), [-0.56, 6.36], contrasts, angle_times, save_path, 'H2O_D2O')
