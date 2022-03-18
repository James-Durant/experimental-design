import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (9,7)
plt.rcParams['figure.dpi'] = 600

import numpy as np
import os

import refnx.dataset, refnx.reflect, refnx.analysis
import periodictable as pt
from base import BaseLipid
from utils import save_plot


class SubphaseSiSiO2(BaseLipid):
    """Defines a model describing a POPC bilayer.

    Attributes:
        name (str): name of the bilayer sample.
        data_path (str): path to directory containing measured data.
        labels (list): label for each measured contrast.
        distances (numpy.ndarray): SLD profile x-axis range.
        scales (list): experimental scale factor for each measured contrast.
        bkgs (list): level of instrument background noise for each contrast.
        dq (float): instrument resolution.
        si_sld (float): SLD of silicon substrate.
        sio2_sld (float): SLD of silicon oxide.
        si_rough (refnx.analysis.Parameter): silicon substrate roughness.
        sio2_thick (refnx.analysis.Parameter): silicon oxide thickness.
        sio2_rough (refnx.analysis.Parameter): silicon oxide roughness.
        sio2_solv (refnx.analysis.Parameter): silicon oxide hydration.
        params (list): varying model parameters.
        structures (list): structures corresponding to each measured contrast.
        objectives (list): objectives corresponding to each measured contrast.

    """
    def __init__(self):
        self.name = 'SiSiO2_subphase'
        self.data_path = os.path.join(os.path.dirname(__file__),
                                      '..',
                                      'data',
                                      'SiSiO2_subphase')

        self.labels = ['Si-D2O', 'Si-H2O']
        self.scales = [1, 1]
        self.bkgs = [3.20559e-06, 3.80358e-06]
        self.dq = 2

        # Define known values.
        self.si_sld      = 2.073
        self.sio2_sld    = 3.41

        # Define the varying parameters of the model.
        self.si_rough = refnx.analysis.Parameter(2, 'Si/SiO2 Roughness', (1, 8))
        self.sio2_thick = refnx.analysis.Parameter(14.7, 'SiO2 Thickness', (5, 20))
        self.sio2_solv = refnx.analysis.Parameter(0.245, 'SiO2 Hydration', (0, 0.8))

        self.params = [self.si_rough,
                       self.sio2_thick,
                       self.sio2_solv]

        self.underlayer_params = [self.si_rough,
                                  self.sio2_thick]

        # Vary all of the parameters defined above.
        for param in self.params:
            param.vary=True

        # Call the BaseLipid constructor.
        super().__init__()

    def _create_objectives(self):
        """Creates objectives corresponding to each measured contrast."""
        # Define scattering length densities of D2O and H2O.
        d2o_sld = 6.19
        h2o_sld = -0.5227

        D2O = refnx.reflect.SLD(d2o_sld)
        H2O = refnx.reflect.SLD(h2o_sld)
        D2O.real.setp(vary=True, bounds=(5.8, 6.35))
        H2O.real.setp(vary=True, bounds=(-0.6, -0.3))

        # Define the layers of the structure.
        substrate = refnx.reflect.SLD(self.si_sld)
        sio2 = refnx.reflect.Slab(self.sio2_thick, self.sio2_sld, self.si_rough, vfsolv=self.sio2_solv)

        # Structure corresponding to measuring the Si/D2O interface.
        si_D2O_structure = substrate | sio2 | D2O(rough=self.si_rough)
        si_H2O_structure = substrate | sio2 | H2O(rough=self.si_rough)

        self.structures = [si_D2O_structure,
                           si_H2O_structure]

        # Iterate over each measured structure.
        self.objectives = []
        for i, structure in enumerate(self.structures):
            # Load the measured data.
            filename = '{}.dat'.format(self.labels[i])
            file_path = os.path.join(self.data_path, filename)
            data = refnx.dataset.ReflectDataset(file_path)

            # Define the model.
            model = refnx.reflect.ReflectModel(structure,
                                               scale=self.scales[i],
                                               #bkg=self.bkgs[i],
                                               bkg=data.y.min(),
                                               dq=self.dq)

            # Combine model and data into an objective that can be fitted.
            self.objectives.append(refnx.analysis.Objective(model, data))

    def _using_conditions(self, contrast_sld, underlayers=None):
        """Creates a structure representing the bilayer measured using
           given measurement conditions.

        Args:
            contrast_sld (float): SLD of contrast to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            refnx.reflect.Structure: bilayer using measurement conditions.

        """
        # Define the layer structure.
        substrate = refnx.reflect.SLD(self.si_sld)
        solution = refnx.reflect.SLD(contrast_sld)(rough=self.sio2_rough)

        sio2 = refnx.reflect.Slab(self.sio2_thick, self.sio2_sld, self.si_rough, vfsolv=self.sio2_solv)
        return substrate | sio2 | solution
