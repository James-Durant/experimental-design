import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = (9,7)
plt.rcParams['figure.dpi'] = 600

import refnx.dataset, refnx.reflect, refnx.analysis
from base import BaseLipid

class BilayerDMPC(BaseLipid):
    """Defines a model describing a symmetric DMPC bilayer.

    Attributes:
        name (str): name of the sample.
        data_path (str): path to directory containing measured data.
        scales (list): experimental scale factor for each measured contrast.
        bkgs (list): level of instrument background noise for each measured contrast.
        dq (float): instrument resolution.
        labels (list): label for each measured contrast.
        distances (numpy.ndarray): SLD profile x-axis range.
        si_sld (float): SLD of silicon substrate.
        sio2_sld (float): SLD of silicon oxide.
        dmpc_hg_vol (float): headgroup volume of DMPC bilayer.
        dmpc_tg_vol (float): tailgroup volume of DMPC bilayer.
        dmpc_hg_sl (float): headgroup scattering length of DMPC bilayer.
        dmpc_tg_sl (float): tailgroup scattering length of DMPC bilayer.
        water_vol (float): water volume of measured system.
        tg_sld (float): tailgroup SLD of DMPC bilayer.
        si_rough (refnx.analysis.Parameter): silicon substrate roughness.
        sio2_thick (refnx.analysis.Parameter): silicon oxide thickness.
        sio2_rough (refnx.analysis.Parameter): silicon oxide roughness.
        sio2_solv (refnx.analysis.Parameter): silicon oxide hydration.
        dmpc_apm (refnx.analysis.Parameter): DMPC area per molecule.
        bilayer_rough (refnx.analysis.Parameter): bilayer roughness.
        bilayer_solv (refnx.analysis.Parameter): bilayer hydration.
        hg_waters (refnx.analysis.Parameter): headgroup bound waters.
        params (list): varying model parameters.
        structures
        objectives

    """
    def __init__(self):
        self.name = 'DMPC_bilayer'
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'DMPC_bilayer')
        self.labels = ['Si-D2O', 'Si-DMPC-D2O', 'Si-DMPC-H2O']
        self.distances = np.linspace(-20, 95, 500)
        
        self.scales = [0.677763, 0.645217, 0.667776]
        self.bkgs = [3.20559e-06, 2.05875e-06, 2.80358e-06]
        self.dq = 2

        # Define known values.
        self.si_sld      = 2.073
        self.sio2_sld    = 3.41
        self.dmpc_hg_vol = 320.9
        self.dmpc_tg_vol = 783.3
        self.dmpc_hg_sl  = 6.41e-4
        self.dmpc_tg_sl  = -3.08e-4
        self.water_vol   = 30.4

        # Calculate the SLD of the tails.
        self.tg_sld = (self.dmpc_tg_sl / self.dmpc_tg_vol) * 1e6

        # Define the varying parameters of the model.
        self.si_rough      = refnx.analysis.Parameter(2,     'Si/SiO2 Roughness',      (1,8))
        self.sio2_thick    = refnx.analysis.Parameter(14.7,  'SiO2 Thickness',         (5,20))
        self.sio2_rough    = refnx.analysis.Parameter(2,     'SiO2/DMPC Roughness',    (1,8))
        self.sio2_solv     = refnx.analysis.Parameter(0.245, 'SiO2 Hydration',         (0,1))
        self.dmpc_apm      = refnx.analysis.Parameter(49.9,  'DMPC Area Per Molecule', (30,60))
        self.bilayer_rough = refnx.analysis.Parameter(6.57,  'Bilayer Roughness',      (1,8))
        self.bilayer_solv  = refnx.analysis.Parameter(0.074, 'Bilayer Hydration',      (0,1))
        self.hg_waters     = refnx.analysis.Parameter(3.59,  'Headgroup Bound Waters', (0,20))

        self.params = [self.si_rough,
                       self.sio2_thick,
                       self.sio2_rough,
                       self.sio2_solv,
                       self.dmpc_apm,
                       self.bilayer_rough,
                       self.bilayer_solv,
                       self.hg_waters]

        # Vary all of the parameters defined above.
        for param in self.params:
            param.vary=True

        # Call the base bilayer class' constructor.
        super().__init__()

    def _create_objectives(self):
        """Creates objectives corresponding to each measured contrast."""
        # Define scattering lengths and densities of D2O and H2O.
        d2o_sl  = 2e-4
        d2o_sld = 6.19
        h2o_sl  = -1.64e-5
        h2o_sld = -0.5227

        D2O = refnx.reflect.SLD(d2o_sld)
        H2O = refnx.reflect.SLD(h2o_sld)

        # Relate headgroup bound waters to scattering lengths and volumes.
        hg_water_d2o_sl = self.hg_waters * d2o_sl
        hg_water_h2o_sl = self.hg_waters * h2o_sl
        hg_water_vol    = self.hg_waters * self.water_vol

        # Add to the headgroup volumes and scattering lengths in both contrast.
        vol_hg = self.dmpc_hg_vol + hg_water_vol

        dmpc_hg_sl_d2o = self.dmpc_hg_sl + hg_water_d2o_sl
        dmpc_hg_sl_h2o = self.dmpc_hg_sl + hg_water_h2o_sl

        # Calculate the SLD of the headgroup in both contrast cases
        sld_hg_d2o = (dmpc_hg_sl_d2o / vol_hg) * 1e6 # SLD = sum b / v
        sld_hg_h2o = (dmpc_hg_sl_h2o / vol_hg) * 1e6

        # Calculate the thickness from the headgroup volume over the lipid area per molecule.
        hg_thick = vol_hg / self.dmpc_apm # Thickness = v / APM

        # Calculate the thickness of the tailgroup
        tg_thick = self.dmpc_tg_vol / self.dmpc_apm

        # Define the layers of the structure.
        substrate = refnx.reflect.SLD(self.si_sld)
        sio2 = refnx.reflect.Slab(self.sio2_thick, self.sio2_sld, self.si_rough, vfsolv=self.sio2_solv)

        inner_hg_d2o = refnx.reflect.Slab(hg_thick, sld_hg_d2o,  self.sio2_rough,    vfsolv=self.bilayer_solv)
        outer_hg_d2o = refnx.reflect.Slab(hg_thick, sld_hg_d2o,  self.bilayer_rough, vfsolv=self.bilayer_solv)
        inner_hg_h2o = refnx.reflect.Slab(hg_thick, sld_hg_d2o,  self.sio2_rough,    vfsolv=self.bilayer_solv)
        outer_hg_h2o = refnx.reflect.Slab(hg_thick, sld_hg_h2o,  self.bilayer_rough, vfsolv=self.bilayer_solv)
        tg           = refnx.reflect.Slab(tg_thick, self.tg_sld, self.bilayer_rough, vfsolv=self.bilayer_solv)

        # Structure corresponding to measuring the Si/D2O interface.
        si_D2O_structure = substrate | sio2 | D2O(rough=self.sio2_rough)

        # Two structures corresponding to each measured contrast.
        si_DMPC_D2O_structure = substrate | sio2 | inner_hg_d2o | tg | tg | outer_hg_d2o | D2O(rough=self.bilayer_rough)
        si_DMPC_H2O_structure = substrate | sio2 | inner_hg_h2o | tg | tg | outer_hg_h2o | H2O(rough=self.bilayer_rough)

        self.structures = [si_D2O_structure, si_DMPC_D2O_structure, si_DMPC_H2O_structure]

        # Label the structures.
        for i, structure in enumerate(self.structures):
            structure.name = self.labels[i]

        # Define models using structures above.
        models = [refnx.reflect.ReflectModel(structure, scale=scale, bkg=bkg, dq=self.dq)
                  for structure, scale, bkg in zip(self.structures, self.scales, self.bkgs)]
        
        # Load the measured datasets.
        datasets = [refnx.dataset.ReflectDataset(os.path.join(self.data_path, '{}.dat'.format(label)))
                    for label in self.labels]

        # Combine models and datasets into objectives that can be fitted.
        self.objectives = [refnx.analysis.Objective(model, data)
                           for model, data in zip(models, datasets)]

    def _using_conditions(self, contrast_sld, underlayers=None):
        """Creates a structure representing the bilayer measured using given measurement conditions.

        Args:
            contrast_sld (float): SLD of contrast to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            refnx.reflect.Structure: structure describing measurement conditions.

        """
        # Calculate the SLD of the headgroup with the given contrast SLD.
        hg_sld = contrast_sld*0.27 + 1.98*0.73

        # Calculate the headgroup and tailgroup thicknesses with the given contrast SLD.
        vol_hg = self.dmpc_hg_vol + self.hg_waters*self.water_vol
        hg_thick = vol_hg / self.dmpc_apm
        tg_thick = self.dmpc_tg_vol / self.dmpc_apm

        substrate = refnx.reflect.SLD(self.si_sld)
        solution = refnx.reflect.SLD(contrast_sld)(rough=self.bilayer_rough)

        inner_hg = refnx.reflect.Slab(hg_thick, hg_sld, self.sio2_rough, vfsolv=self.bilayer_solv)
        outer_hg = refnx.reflect.Slab(hg_thick, hg_sld, self.bilayer_rough, vfsolv=self.bilayer_solv)
        tg = refnx.reflect.Slab(tg_thick, self.tg_sld, self.bilayer_rough, vfsolv=self.bilayer_solv)

        # Add the underlayer if specified.
        if underlayers is None:
            sio2 = refnx.reflect.Slab(self.sio2_thick, self.sio2_sld, self.si_rough, vfsolv=self.sio2_solv)
            return substrate | sio2 | inner_hg | tg | tg | outer_hg | solution
        else:
            # Add each underlayer with given thickness and SLD.
            sio2 = refnx.reflect.Slab(self.sio2_thick, self.sio2_sld, self.si_rough, vfsolv=0)
            structure = substrate | sio2
            for thick, sld in underlayers:
                underlayer = refnx.reflect.SLD(sld)(thick, 2)
                structure |= underlayer
            return structure | inner_hg | tg | tg | outer_hg | solution

    def underlayer_info(self, angle_times, contrasts, underlayers):
        """Calculates the Fisher information matrix for a lipid sample with `underlayers`,
           and with contrasts measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        params_all = self.params
        self.params = [self.si_rough,
                       self.sio2_thick,
                       self.dmpc_apm,
                       self.bilayer_rough,
                       self.bilayer_solv,
                       self.hg_waters]
        
        g = super().underlayer_info(angle_times, contrasts, underlayers)
        self.params = params_all
        return g

class BilayerDPPC(BaseLipid):
    """Defines a model describing an asymmetric bilayer defined by a single
       asymmetry value.

    Attributes:
        name
        labels (list): label for each measured contrast.
        data_path (str): path to directory containing measured data.
        distances (numpy.ndarray): SLD profile x-axis range.
        contrast_slds (list): SLD of each measured contrast.
        scale (float): experimental scale factor for measured contrasts.
        bkgs (list): level of instrument background noise for each measured contrast.
        dq (float): instrument resolution.
        si_sld (float): SLD of silicon substrate.
        sio2_sld (float): SLD of silicon oxide.
        pc_hg_sld (float):
        dPC_tg (float):
        hLPS_tg (float):
        core_D2O (float):
        core_H2O (float):
        si_rough (refnx.analysis.Parameter): silicon substrate roughness.
        sio2_thick (refnx.analysis.Parameter): silicon oxide thickness.
        sio2_rough (refnx.analysis.Parameter): silicon oxide roughness.
        sio2_solv (refnx.analysis.Parameter): silicon oxide hydration.
        inner_hg_thick (refnx.analysis.Parameter): inner headgroup thickness.
        inner_hg_solv (refnx.analysis.Parameter): inner headgroup hydration.
        bilayer_rough (refnx.analysis.Parameter): bilayer roughness.
        inner_tg_thick (refnx.analysis.Parameter): inner tailgroup thickness.
        outer_tg_thick (refnx.analysis.Parameter): outer tailgroup thickness.
        tg_solv (refnx.analysis.Parameter): tailgroup hydration.
        core_thick (refnx.analysis.Parameter): core thickness.
        core_solv (refnx.analysis.Parameter): core hydration.
        asym_value
        params (list): varying model parameters.
        structures
        objectives

    """
    def __init__(self):
        self.name = 'DPPC_RaLPS_bilayer'
        self.labels = ['dDPPC-RaLPS-D2O', 'dDPPC-RaLPS-SMW', 'dDPPC-RaLPS-H2O']
        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'DPPC_RaLPS_bilayer')
        self.distances = np.linspace(-30, 110, 500)
        
        self.contrast_slds = [6.14, 2.07, -0.56]
        self.scale = 0.8
        self.bkgs = [4.6e-6, 8.6e-6, 8.7e-6]
        self.dq = 4

        # Define known values.
        self.si_sld    =  2.07
        self.sio2_sld  =  3.41
        self.pc_hg_sld =  1.98
        self.dPC_tg    =  7.45
        self.hLPS_tg   = -0.37
        self.core_D2O  =  4.2
        self.core_H2O  =  2.01

        # Define the varying parameters of the model.
        self.si_rough       = refnx.analysis.Parameter(5.5,    'Si/SiO2 Roughness',         (3,8))
        self.sio2_thick     = refnx.analysis.Parameter(13.4,   'SiO2 Thickness',            (10,30))
        self.sio2_rough     = refnx.analysis.Parameter(3.2,    'SiO2/Bilayer Roughness',    (2,5))
        self.sio2_solv      = refnx.analysis.Parameter(0.038,  'SiO2 Hydration',            (0,0.5))
        self.inner_hg_thick = refnx.analysis.Parameter(9.0,    'Inner Headgroup Thickness', (5,20))
        self.inner_hg_solv  = refnx.analysis.Parameter(0.39,   'Inner Headgroup Hydration', (0,1))
        self.bilayer_rough  = refnx.analysis.Parameter(4.0,    'Bilayer Roughness',         (0,12))
        self.inner_tg_thick = refnx.analysis.Parameter(16.7,   'Inner Tailgroup Thickness', (10,20))
        self.outer_tg_thick = refnx.analysis.Parameter(14.9,   'Outer Tailgroup Thickness', (10,20))
        self.tg_solv        = refnx.analysis.Parameter(0.0085, 'Tailgroup Hydration',       (0,1))
        self.core_thick     = refnx.analysis.Parameter(28.7,   'Core Thickness',            (0,50))
        self.core_solv      = refnx.analysis.Parameter(0.26,   'Core Hydration',            (0,1))
        self.asym_value     = refnx.analysis.Parameter(0.95,   'Asymmetry Value',           (0,1))
        
        self.params = [self.si_rough,
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
        for param in self.params:
            param.vary=True
            
        self._create_objectives()

    def _create_objectives(self):
        """Creates objectives corresponding to each measured contrast."""
        # Define structures for each contrast.
        self.structures = [self._using_conditions(contrast_sld) for contrast_sld in self.contrast_slds]

        # Label each structure.
        for i, label in enumerate(self.labels):
            self.structures[i].name = label

        # Define models for the structures above.
        models = [refnx.reflect.ReflectModel(structure, scale=self.scale, bkg=bkg, dq=self.dq)
                  for structure, bkg in list(zip(self.structures, self.bkgs))]

        # Load the measured data for each contrast.
        datasets = [refnx.dataset.ReflectDataset(os.path.join(self.data_path, '{}.dat'.format(label)))
                    for label in self.labels]

        # Combine models and datasets into objectives corresponding to each contrast.
        self.objectives = [refnx.analysis.Objective(model, data) for model, data in list(zip(models, datasets))]

    def _using_conditions(self, contrast_sld, underlayers=None):
        """Creates a structure representing the bilayer measured using given measurement conditions.

        Args:
            contrast_sld (float): SLD of contrast to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            refnx.reflect.Structure: structure describing measurement conditions.

        """
        # Calculate core SLD with the given contrast SLD.
        contrast_point = (contrast_sld + 0.56) / (6.35 + 0.56)
        core_sld = contrast_point*self.core_D2O + (1-contrast_point)*self.core_H2O

        # Use asymmetry to define inner and outer tailgroup SLDs.
        inner_tg_sld = self.asym_value*self.dPC_tg + (1-self.asym_value)*self.hLPS_tg
        outer_tg_sld = self.asym_value*self.hLPS_tg + (1-self.asym_value)*self.dPC_tg

        substrate = refnx.reflect.SLD(self.si_sld)
        solution  = refnx.reflect.SLD(contrast_sld)(0, self.bilayer_rough)

        inner_hg = refnx.reflect.Slab(self.inner_hg_thick, self.pc_hg_sld, self.sio2_rough,    vfsolv=self.inner_hg_solv)
        inner_tg = refnx.reflect.Slab(self.inner_tg_thick, inner_tg_sld,   self.bilayer_rough, vfsolv=self.tg_solv)
        outer_tg = refnx.reflect.Slab(self.outer_tg_thick, outer_tg_sld,   self.bilayer_rough, vfsolv=self.tg_solv)
        core     = refnx.reflect.Slab(self.core_thick,     core_sld,       self.bilayer_rough, vfsolv=self.core_solv)

        # Add the underlayer if specified.
        if underlayers is None:
            sio2 = refnx.reflect.Slab(self.sio2_thick, self.sio2_sld, self.si_rough, vfsolv=self.sio2_solv)
            return substrate | sio2 | inner_hg | inner_tg | outer_tg | core | solution
        else:
            # Add each underlayer with given thickness and SLD.
            sio2 = refnx.reflect.Slab(self.sio2_thick, self.sio2_sld, self.si_rough, vfsolv=0)
            structure = substrate | sio2
            for thick, sld in underlayers:
                underlayer = refnx.reflect.SLD(sld)(thick, 2)
                structure |= underlayer
            return structure | inner_hg | inner_tg | outer_tg | core | solution

    def underlayer_info(self, angle_times, contrasts, underlayers):
        """Calculates the Fisher information matrix for a lipid sample with `underlayers`,
           and with contrasts measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        params_all = self.params
        self.params = [self.si_rough,
                       self.sio2_thick,
                       self.inner_hg_thick,
                       self.inner_hg_solv,
                       self.bilayer_rough,
                       self.inner_tg_thick,
                       self.outer_tg_thick,
                       self.tg_solv,
                       self.core_thick,
                       self.core_solv]
        
        g = super().underlayer_info(angle_times, contrasts, underlayers)
        self.params = params_all
        return g
    
if __name__ == '__main__':
    save_path = '../results'

    dmpc_bilayer = BilayerDMPC() 
    dmpc_bilayer.sld_profile(save_path)    
    dmpc_bilayer.reflectivity_profile(save_path)

    dppc_bilayer = BilayerDPPC() 
    dppc_bilayer.sld_profile(save_path)    
    dppc_bilayer.reflectivity_profile(save_path)

    plt.close('all')