import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = (9,7)
plt.rcParams['figure.dpi'] = 600

from abc import ABC, abstractmethod

import refnx.dataset, refnx.reflect, refnx.analysis
import refl1d.material, refl1d.model, refl1d.probe, refl1d.experiment, refl1d.magnetism
import bumps.parameter, bumps.fitproblem

from simulate import simulate, simulate_magnetic, refl1d_experiment, reflectivity
from utils import fisher, Sampler, save_plot

class VariableAngle(ABC):
    """Abstract class representing whether the measurement angle of a sample can be varied."""
    @abstractmethod
    def angle_info(self, angle_times):
        """Calculates the Fisher information matrix for a sample measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.

        """
        pass

class VariableContrast(ABC):
    """Abstract class representing whether the contrast of a sample can be varied."""
    @abstractmethod
    def contrast_info(self, angle_times, contrasts):
        """Calculates the Fisher information matrix for a sample with contrasts
           measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.

        """
        pass

class VariableUnderlayer(ABC):
    """Abstract class representing whether the underlayer(s) of a sample can be varied."""
    @abstractmethod
    def underlayer_info(self, angle_times, contrasts, underlayers):
        """Calculates the Fisher information matrix for a sample with `underlayers`,
           and with contrasts measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (tuple): thickness and SLD of each underlayer to add.

        """
        pass

class BaseSample(VariableAngle):
    """Abstract class representing a typical neutron reflectometry sample."""
    @abstractmethod
    def sld_profile(self, save_path):
        """Plots the SLD profile of the sample.

        Args:
            save_path (str): path to directory to save SLD profile to.

        """
        pass

    @abstractmethod
    def reflectivity_profile(self, save_path):
        """Plots the reflectivity profile of the sample.

        Args:
            save_path (str): path to directory to save reflectivity profile to.

        """
        pass

    @abstractmethod
    def nested_sampling(self, angle_times, save_path, filename, dynamic=False):
        """Runs nested sampling on simulated data of the sample.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            save_path (str): path to directory to save corner plot to.
            filename (str): name of file to save corner plot to.
            dynamic (bool): whether to use static or dynamic nested sampling.

        """
        pass

class Sample(BaseSample):
    """Wrapper class for a standard refnx or Refl1D reflectometry sample.

    Attributes:
        structure (refnx.reflect.Structure or refl1d.model.Stack): refnx or Refl1D sample.
        name (str): name of the sample.
        params (list): varying parameters of sample.

    """
    def __init__(self, structure):
        self.structure = structure
        self.name = structure.name
        self.params = Sample.__vary_structure(structure)

    @staticmethod
    def __vary_structure(structure, bound_size=0.2):
        """Varies the SLD and thickness of each layer of a given `structure`.

        Args:
            structure (refnx.reflect.Structure or refl1d.model.Stack): structure to vary.
            bound_size (float): size of bounds to place on varying parameters.

        Returns:
            list: varying parameters of sample.

        """
        params = []
        # The structure was defined in refnx.
        if isinstance(structure, refnx.reflect.Structure):
            # Vary the SLD and thickness of each component (layer).
            for component in structure[1:-1]:
                sld = component.sld.real
                sld_bounds = (sld.value*(1-bound_size), sld.value*(1+bound_size))
                sld.setp(vary=True, bounds=sld_bounds)
                params.append(sld)

                thick = component.thick
                thick_bounds = (thick.value*(1-bound_size), thick.value*(1+bound_size))
                thick.setp(vary=True, bounds=thick_bounds)
                params.append(thick)

        # The structure was defined in Refl1D.
        elif isinstance(structure, refl1d.model.Stack):
            # Vary the SLD and thickness of each component (layer).
            for component in structure[1:-1]:
                sld = component.material.rho
                sld.pmp(bound_size*100)
                params.append(sld)

                thick = component.thickness
                thick.pmp(bound_size*100)
                params.append(thick)

        # Otherwise the structure is invalid.
        else:
            raise RuntimeError('invalid structure given')

        return params

    def angle_info(self, angle_times, contrasts=None):
        """Calculates the Fisher information matrix for a sample measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        model, data = simulate(self.structure, angle_times)
        qs, counts, models = [data[:,0]], [data[:,3]], [model]
        return fisher(qs, self.params, counts, models)

    def sld_profile(self, save_path):
        """Plots the SLD profile of the sample.

        Args:
            save_path (str): path to directory to save SLD profile to.

        """
        # Currently not defined for Refl1D samples.
        if isinstance(self.structure, refnx.reflect.Structure):
            z, slds = self.structure.sld_profile()
            
        elif isinstance(self.structure, refl1d.model.Stack):
            q = np.geomspace(0.005, 0.3, 500)
            scale, bkg, dq = 1, 1e-6, 2
            experiment = refl1d_experiment(self.structure, q, scale, bkg, dq)
            z, slds, _ = experiment.smooth_profile()
            
        else:
            raise RuntimeError('invalid structure given')

        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the SLD profile.
        ax.plot(z, slds, color='black', label=self.name)

        ax.set_xlabel('$\mathregular{Distance\ (\AA)}$', fontsize=11, weight='bold')
        ax.set_ylabel('$\mathregular{SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'sld_profile')

    def reflectivity_profile(self, save_path, q_min=0.005, q_max=0.3, points=500, scale=1, bkg=1e-7, dq=2):
        """Plots the reflectivity profile of the sample.

        Args:
            save_path (str): path to directory to save reflectivity profile to.
            q_min (float): minimum Q value to plot.
            q_max (float): maximum Q value to plot.
            points (int): number of points to plot.
            scale (float): experimental scale factor.
            bkg (float): level of instrument background noise.
            dq (float): instrument resolution.

        """
        q = np.geomspace(q_min, q_max, points)
        if isinstance(self.structure, refnx.reflect.Structure):
            model = refnx.reflect.ReflectModel(self.structure, scale=scale, bkg=bkg, dq=dq)
            
        elif isinstance(self.structure, refl1d.model.Stack):
            model = refl1d_experiment(self.structure, q, scale, bkg, dq)
            
        else:
            raise RuntimeError('invalid structure given')

        r = reflectivity(q, model) # Calculate the model reflectivity.

        # Plot the model reflectivity against Q.
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(q, r, color='black')

        ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
        ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
        ax.set_yscale('log')

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'reflectivity_profile')

    def nested_sampling(self, angle_times, save_path, filename, dynamic=False):
        """Runs nested sampling on simulated data of the sample.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            save_path (str): path to directory to save corner plot to.
            filename (str): name of file to save corner plot to.
            dynamic (bool): whether to use static or dynamic nested sampling.

        """
        model, data = simulate(self.structure, angle_times)

        if isinstance(self.structure, refnx.reflect.Structure):
            dataset = refnx.reflect.ReflectDataset([data[:,0], data[:,1], data[:,2]])
            objective = refnx.anaylsis.Objective(model, dataset)

        elif isinstance(self.structure, refl1d.model.Stack):
            objective = bumps.fitproblem.FitProblem(model)
            
        else:
            raise RuntimeError('invalid structure given')
            
        # Sample the objective using nested sampling.
        sampler = Sampler(objective)
        fig = sampler.sample(dynamic=dynamic)

        # Save the sampling corner plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, filename+'_nested_sampling')

class MagneticSample(BaseSample):
    def _set_dq(self, probe):
        # Transform the resolution from refnx to Refl1D format.
        dq = self.dq / (100*np.sqrt(8*np.log(2)))
    
        q_array = probe.Q
    
        # Calculate the dq array and use it to define a Q probe.
        dq_array = probe.Q * dq
        probe.dQ = dq_array
    
        # Adjust probe calculation for constant resolution.
        argmin, argmax = np.argmin(q_array), np.argmax(q_array)
        probe.calc_Qo = np.linspace(q_array[argmin] - 3.5*dq_array[argmin],
                                    q_array[argmax] + 3.5*dq_array[argmax],
                                    21*len(q_array))
   
    def sld_profile(self, save_path):
        """Plots the SLD profile of the sample.

        Args:
            save_path (str): path to directory to save SLD profile to.

        """
        q = np.geomspace(0.005, 0.3, 500)
        scale, bkg, dq = 1, 1e-6, 2
        experiment = refl1d_experiment(self.structure, q, scale, bkg, dq, 0)
        z, slds, _, slds_mag, _ = experiment.magnetic_smooth_profile()
            
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the SLD profile.
        ax.plot(z, slds, color='blue', label='SLD')
        ax.plot(z, slds_mag, color='green', label='Magnetic SLD')

        ax.set_xlabel('$\mathregular{Distance\ (\AA)}$', fontsize=11, weight='bold')
        ax.set_ylabel('$\mathregular{SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
        ax.legend()
        
        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'sld_profile')
   
    def reflectivity_profile(self, save_path):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        labels = ['++', '+-', '-+', '--']
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        probes = self.experiment.probe.xs
        count = 0
        for i, qr in enumerate(self.experiment.reflectivity()):
            if qr is not None:
                probe = probes[i]
                ax.errorbar(probe.Q, probe.R, probe.dR, marker='o', ms=2, lw=0, elinewidth=0.5, capsize=0.5, label=labels[i]+' Data', color=colours[count])
                ax.plot(probe.Q, qr[1], color=colours[count], zorder=20, label=labels[i]+' Fitted')
                count += 1
    
        ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
        ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
        ax.set_yscale('log')
        ax.legend()

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'reflectivity_profile')     

    def nested_sampling(self, angle_times, save_path, filename, dynamic=False):
        """Runs nested sampling on simulated data of the sample.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            save_path (str): path to directory to save corner plot to.
            filename (str): name of file to save corner plot to.
            dynamic (bool): whether to use static or dynamic nested sampling.

        """
        models, _ = simulate_magnetic(self.structure, angle_times)
        objective = bumps.fitproblem.FitProblem(models)
  
        # Sample the objective using nested sampling.
        sampler = Sampler(objective)
        fig = sampler.sample(dynamic=dynamic)

        # Save the sampling corner plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, filename+'_nested_sampling')

class YIG(MagneticSample, VariableUnderlayer):
    def __init__(self):
        self.name = 'YIG_sample'
        
        self.data_path = '../experimental-design/data/YIG_sample'
        self.scale = 1.025
        self.bkg = 4e-7
        self.dq = 2.8
        self.mag_angle = 90

        self.Pt_sld   = bumps.parameter.Parameter(5.590, name='Pt SLD')
        self.Pt_thick = bumps.parameter.Parameter(22.1,  name='Pt Thickness')
        self.Pt_rough = bumps.parameter.Parameter(8.11,  name='Pt/Air Roughness')
        self.Pt_mag   = bumps.parameter.Parameter(0.019, name='Pt Magnetic SLD')
        
        self.FePt_sld   = bumps.parameter.Parameter(4.717, name='FePt SLD')
        self.FePt_thick = bumps.parameter.Parameter(18.1,  name='FePt Thickness')
        self.FePt_rough = bumps.parameter.Parameter(0.02,  name='FePt\Pt Roughness')
        
        self.YIG_sld   = bumps.parameter.Parameter(5.810, name='YIG SLD')
        self.YIG_thick = bumps.parameter.Parameter(706.5, name='YIG Thickness')
        self.YIG_rough = bumps.parameter.Parameter(11.8,  name='YIG\FePt Roughness')
        self.YIG_mag   = bumps.parameter.Parameter(0.373, name='YIG Magnetic SLD')
        
        self.sub_sld   = bumps.parameter.Parameter(4.994, name='Substrate SLD')
        self.sub_rough = bumps.parameter.Parameter(88.5,  name='Substrate\YIG Roughness')
        
        self.params = [self.Pt_sld,
                       self.Pt_thick,
                       self.Pt_rough,
                       self.Pt_mag,
                       self.FePt_sld,
                       self.FePt_thick,
                       self.FePt_rough,
                       self.YIG_sld,
                       self.YIG_thick,
                       self.YIG_rough,
                       self.YIG_mag,
                       self.sub_sld,
                       self.sub_rough]
        
        for param in self.params:
            param.pmp(20)
        
        air = refl1d.material.SLD(rho=0, name='Air')
        Pt = refl1d.material.SLD(rho=self.Pt_sld, name='Pt')(self.Pt_thick, self.Pt_rough, magnetism=refl1d.magnetism.Magnetism(rhoM=self.Pt_mag, thetaM=self.mag_angle))
        FePt = refl1d.material.SLD(rho=self.FePt_sld, name='FePt')(self.FePt_thick, self.FePt_rough)
        YIG = refl1d.material.SLD(rho=self.YIG_sld, name='YIG')(self.YIG_thick, self.YIG_rough, magnetism=refl1d.magnetism.Magnetism(rhoM=self.YIG_mag, thetaM=self.mag_angle))
        sub = refl1d.material.SLD(rho=self.sub_sld, name='Substrate')(0, self.sub_rough)
        
        self.structure = sub | YIG | FePt | Pt | air
        
        self.__create_experiment()

    def __create_experiment(self):
        pp = refl1d.probe.load4(os.path.join(self.data_path, 'YAG_2_Air.u'), sep='\t', intensity=self.scale, background=self.bkg, columns='Q R dR dQ')
        mm = refl1d.probe.load4(os.path.join(self.data_path, 'YAG_2_Air.d'), sep='\t', intensity=self.scale, background=self.bkg, columns='Q R dR dQ')
        pm = None
        mp = None
        
        self._set_dq(pp)
        self._set_dq(mm)
        
        probe = refl1d.probe.PolarizedQProbe(xs=(pp, pm, mp, mm), name='Probe')
        self.experiment = refl1d.experiment.Experiment(sample=self.structure, probe=probe)
    
    def angle_info(self, angle_times):
        models, datasets = simulate_magnetic(sample, angle_times, pp=True, pm=False, mp=False, mm=True)

        qs = [data[:,0] for data in datasets]
        counts = [data[:,3] for data in datasets]

        return fisher(qs, self.params, counts, models)
    
    def underlayer_info(self):
        pass

class BaseBilayer(BaseSample, VariableContrast, VariableUnderlayer):
    """Abstract class representing the base class for a bilayer model."""
    def __init__(self):
        self._create_objectives()

    @abstractmethod
    def _create_objectives(self):
        """Loads the measured data for the bilayer sample."""
        pass

    def angle_info(self, angle_times, contrasts):
        """Calculates the Fisher information matrix for a bilayer sample measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, None)

    def contrast_info(self, angle_times, contrasts):
        """Calculates the Fisher information matrix for a bilayer sample with contrasts
           measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, None)

    def underlayer_info(self, angle_times, contrasts, underlayers):
        """Calculates the Fisher information matrix for a bilayer sample with `underlayers`,
           and with contrasts measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, underlayers)

    def __conditions_info(self, angle_times, contrasts, underlayers):
        """Calculates the Fisher information matrix for a bilayer sample with given conditions.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        qs, counts, models = [], [], []
        for contrast in contrasts:
            model, data = simulate(self._using_conditions(contrast, underlayers), angle_times)

            qs.append(data[:,0])
            counts.append(data[:,3])
            models.append(model)

        return fisher(qs, self.params, counts, models)

    @abstractmethod
    def _using_conditions(self, contrast, underlayers):
        """Creates a refnx structure describing the given measurement conditions.

        Args:
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            refnx.reflect.Structure: structure describing measurement conditions.

        """
        pass

    def sld_profile(self, save_path):
        """Plots the SLD profile of the bilayer sample.

        Args:
            save_path (str): path to directory to save SLD profile to.

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the SLD profile for each measured contrast.
        for structure in self.structures:
            ax.plot(*structure.sld_profile(self.distances))

        ax.set_xlabel('$\mathregular{Distance\ (\AA)}$', fontsize=11, weight='bold')
        ax.set_ylabel('$\mathregular{SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
        ax.legend(self.labels)

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'sld_profile')

    def reflectivity_profile(self, save_path):
        """Plots the reflectivity profile of the bilayer sample.

        Args:
            save_path (str): path to directory to save reflectivity profile to.

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Iterate over each measured contrast.
        for i, objective in enumerate(self.objectives):
            # Get the measured data and calculate the model reflectivity at each point.
            q, r, dr = objective.data.x, objective.data.y, objective.data.y_err
            r_model = objective.model(q)

            # Offset the data (for clarity).
            offset = 10**(-2*i)
            r *= offset
            dr *= offset
            r_model *= offset

            label = self.labels[i]
            if offset != 1:
                label += ' $\mathregular{(x10^{-'+str(2*i)+'})}$'

            # Plot the measured data and the model reflectivity.
            ax.errorbar(q, r, dr, marker='o', ms=3, lw=0, elinewidth=1, capsize=1.5, label=label)
            ax.plot(q, r_model, color='red', zorder=20)

        ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
        ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-10, 2)
        ax.legend()

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'reflectivity_profile')

    def nested_sampling(self, contrasts, angle_times, save_path, filename, underlayers=None, dynamic=False):
        """Runs nested sampling on simulated data of the bilayer sample.

        Args:
            contrasts (list): SLDs of contrasts to simulate.
            angle_times (list): points and counting times for each measurement angle to simulate.
            save_path (str): path to directory to save corner plot to.
            filename (str): name of file to save corner plot to.
            underlayers (list): thickness and SLD of each underlayer to add.
            dynamic (bool): whether to use static or dynamic nested sampling.

        """
        # Create objectives for each contrast to sample with.
        objectives = []
        for contrast in contrasts:
            # Simulate an experiment using the given contrast.
            model, data = simulate(self._using_conditions(contrast, underlayers), angle_times)
            dataset = refnx.dataset.ReflectDataset([data[:,0], data[:,1], data[:,2]])
            objectives.append(refnx.analysis.Objective(model, dataset))

        # Combine objectives into a single global objective.
        global_objective = refnx.analysis.GlobalObjective(objectives)
        global_objective.varying_parameters = lambda: self.params

        # Sample the objective using nested sampling.
        sampler = Sampler(global_objective)
        fig = sampler.sample(dynamic=dynamic)

        # Save the sampling corner plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, filename+'_nested_sampling')

class SymmetricBilayer(BaseBilayer):
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

    """
    def __init__(self):
        self.name = 'symmetric_bilayer'
        self.data_path = '../experimental-design/data/symmetric_bilayer'
        self.scales = [0.677763, 0.645217, 0.667776]
        self.bkgs = [3.20559e-06, 2.05875e-06, 2.80358e-06]
        self.dq = 2
        self.labels = ['Si-D2O', 'Si-DMPC-D2O', 'Si-DMPC-H2O']
        self.distances = np.linspace(-20, 95, 500)

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
        self.models = [refnx.reflect.ReflectModel(structure, scale=scale, bkg=bkg, dq=self.dq)
                       for structure, scale, bkg in list(zip(self.structures, self.scales, self.bkgs))]

        # Load the measured datasets.
        self.datasets = [refnx.dataset.ReflectDataset(os.path.join(self.data_path, '{}.dat'.format(label)))
                         for label in self.labels]

        # Combine models and datasets into objectives that can be fitted.
        self.objectives = [refnx.analysis.Objective(model, data) for model, data in list(zip(self.models, self.datasets))]

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
        solution  = refnx.reflect.SLD(contrast_sld)(rough=self.bilayer_rough)

        sio2     = refnx.reflect.Slab(self.sio2_thick, self.sio2_sld, self.si_rough,      vfsolv=self.sio2_solv)
        inner_hg = refnx.reflect.Slab(hg_thick,        hg_sld,        self.sio2_rough,    vfsolv=self.bilayer_solv)
        outer_hg = refnx.reflect.Slab(hg_thick,        hg_sld,        self.bilayer_rough, vfsolv=self.bilayer_solv)
        tg       = refnx.reflect.Slab(tg_thick,        self.tg_sld,   self.bilayer_rough, vfsolv=self.bilayer_solv)

        # Add the underlayer if specified.
        if underlayers is None:
            return substrate | sio2 | inner_hg | tg | tg | outer_hg | solution
        else:
            # Add each underlayer with given thickness and SLD.
            structure = substrate | sio2
            for thick, sld in underlayers:
                underlayer = refnx.reflect.SLD(sld)(thick, self.sio2_rough, self.sio2_solv)
                structure |= underlayer
            return structure | inner_hg | tg | tg | outer_hg | solution

class AsymmetricBilayer(BaseBilayer):
    """Defines a model describing an asymmetric bilayer. This model can either
       be defined using single or double asymmetry: the implementation for
       which is provided in the classes that inherit from this parent class.

    Attributes:
        data_path (str): path to directory containing measured data.
        contrast_slds (list): SLD of each measured contrast.
        scale (float): experimental scale factor for measured contrasts.
        bkgs (list): level of instrument background noise for each measured contrast.
        dq (float): instrument resolution.
        labels (list): label for each measured contrast.
        distances (numpy.ndarray): SLD profile x-axis range.
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
        params (list): varying model parameters.

    """
    def __init__(self):
        self.data_path = '../experimental-design/data/asymmetric_bilayer'
        self.contrast_slds = [6.14, 2.07, -0.56]
        self.scale = 0.8
        self.bkgs = [4.6e-6, 8.6e-6, 8.7e-6]
        self.dq = 4
        self.labels = ['dPC-RaLPS-D2O', 'dPC-RaLPS-SMW', 'dPC-RaLPS-H2O']
        self.distances = np.linspace(-30, 110, 500)

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
                       self.core_solv]

        # Vary all of the parameters defined above.
        for param in self.params:
            param.vary=True

    def _create_objectives(self):
        """Creates objectives corresponding to each measured contrast."""
        # Define structures for each contrast.
        self.structures = [self._using_conditions(contrast_sld) for contrast_sld in self.contrast_slds]

        # Label each structure.
        for i, label in enumerate(self.labels):
            self.structures[i].name = label

        # Define models for the structures above.
        self.models = [refnx.reflect.ReflectModel(structure, scale=self.scale, bkg=bkg, dq=self.dq)
                       for structure, bkg in list(zip(self.structures, self.bkgs))]

        # Load the measured data for each contrast.
        self.datasets = [refnx.dataset.ReflectDataset(os.path.join(self.data_path, '{}.dat'.format(label)))
                         for label in self.labels]

        # Combine models and datasets into objectives corresponding to each contrast.
        self.objectives = [refnx.analysis.Objective(model, data) for model, data in list(zip(self.models, self.datasets))]

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

        substrate = refnx.reflect.SLD(self.si_sld)
        solution  = refnx.reflect.SLD(contrast_sld)(0, self.bilayer_rough)

        # Make sure this method is not being called from this class (should be from child class).
        if not hasattr(self, 'inner_tg_sld') or not hasattr(self, 'outer_tg_sld'):
            raise RuntimeError('inner/outer tailgroup SLD not defined')

        sio2     = refnx.reflect.Slab(self.sio2_thick,     self.sio2_sld,     self.si_rough,      vfsolv=self.sio2_solv)
        inner_hg = refnx.reflect.Slab(self.inner_hg_thick, self.pc_hg_sld,    self.sio2_rough,    vfsolv=self.inner_hg_solv)
        inner_tg = refnx.reflect.Slab(self.inner_tg_thick, self.inner_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        outer_tg = refnx.reflect.Slab(self.outer_tg_thick, self.outer_tg_sld, self.bilayer_rough, vfsolv=self.tg_solv)
        core     = refnx.reflect.Slab(self.core_thick,     core_sld,          self.bilayer_rough, vfsolv=self.core_solv)

        # Add the underlayer if specified.
        if underlayers is None:
            return substrate | sio2 | inner_hg | inner_tg | outer_tg | core | solution
        else:
            # Add each underlayer with given thickness and SLD.
            structure = substrate | sio2
            for thick, sld in underlayers:
                underlayer = refnx.reflect.SLD(sld)(thick, self.sio2_rough, self.sio2_solv)
                structure |= underlayer
            return structure | inner_hg | inner_tg | outer_tg | core | solution

class SingleAsymmetricBilayer(AsymmetricBilayer):
    """Defines a model describing an asymmetric bilayer defined by a single
       asymmetry value. Inherits all of the attributes of the parent class.

    Attributes:
        name (str): name of the sample.
        asym_value (refnx.analysis.Parameter): bilayer asymmetry.
        inner_tg_sld (refnx.reflect.SLD): inner tailgroup SLD.
        outer_tg_sld (refnx.reflect.SLD): outer tailgroup SLD.

    """
    def __init__(self):
        super().__init__() # Call parent class' constructor.
        self.name = 'single_asymmetric_bilayer'

        # Define the single asymmetry parameter.
        self.asym_value = refnx.analysis.Parameter(0.95, 'Asymmetry Value', (0,1), True)
        self.params.append(self.asym_value)

        # Use asymmetry to define inner and outer tailgroup SLDs.
        self.inner_tg_sld = refnx.reflect.SLD(self.asym_value*self.dPC_tg + (1-self.asym_value)*self.hLPS_tg)
        self.outer_tg_sld = refnx.reflect.SLD(self.asym_value*self.hLPS_tg + (1-self.asym_value)*self.dPC_tg)

        # Load the measured data for the sample.
        self._create_objectives()

class DoubleAsymmetricBilayer(AsymmetricBilayer):
    """Defines a model describing an asymmetric bilayer defined by two
       asymmetry values. Inherits all of the attributes of the parent class.

    Attributes:
        name (str): name of the sample.
        inner_tg_pc (refnx.analysis.Parameter): 1st bilayer asymmetry.
        outer_tg_pc (refnx.analysis.Parameter): 2nd bilayer asymmetry.
        inner_tg_sld (refnx.reflect.SLD): inner tailgroup SLD.
        outer_tg_sld (refnx.reflect.SLD): outer tailgroup SLD.

    """
    def __init__(self):
        super().__init__() # Call parent class constructor.
        self.name = 'double_asymmetric_bilayer'

        # Define the two asymmetry parameters.
        self.inner_tg_pc = refnx.analysis.Parameter(0.95, 'Inner Tailgroup PC', (0,1), True)
        self.outer_tg_pc = refnx.analysis.Parameter(0.063, 'Outer Tailgroup PC', (0,1), True)
        self.params.append(self.inner_tg_pc)
        self.params.append(self.outer_tg_pc)

        # Use the asymmetry parameters to define inner and outer tailgroup SLDs.
        self.inner_tg_sld = refnx.reflect.SLD(self.inner_tg_pc*self.dPC_tg + (1-self.inner_tg_pc)*self.hLPS_tg)
        self.outer_tg_sld = refnx.reflect.SLD(self.outer_tg_pc*self.dPC_tg + (1-self.outer_tg_pc)*self.hLPS_tg)

        # Load the measured data for the sample.
        self._create_objectives()

def simple_sample():
    """Defines a simple sample.

    Returns:
        structures.Sample: structure in format for information calculation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(4, name='Layer 1')(thick=100, rough=2)
    layer2 = refnx.reflect.SLD(8, name='Layer 2')(thick=150, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | substrate
    structure.name = 'simple_sample'
    return Sample(structure)

def many_param_sample():
    """Defines a sample with many parameters.

    Returns:
        structures.Sample: structure in format for information calculation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(2.0, name='Layer 1')(thick=50, rough=6)
    layer2 = refnx.reflect.SLD(1.7, name='Layer 2')(thick=15, rough=2)
    layer3 = refnx.reflect.SLD(0.8, name='Layer 3')(thick=60, rough=2)
    layer4 = refnx.reflect.SLD(3.2, name='Layer 4')(thick=40, rough=2)
    layer5 = refnx.reflect.SLD(4.0, name='Layer 5')(thick=18, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | layer3 | layer4 | layer5 | substrate
    structure.name = 'many_param_sample'
    return Sample(structure)

def thin_layer_sample_1():
    """Defines a 2-layer sample with thin layers.

    Returns:
        structures.Sample: structure in format for information calculation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(4, name='Layer 1')(thick=200, rough=2)
    layer2 = refnx.reflect.SLD(6, name='Layer 2')(thick=6, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | substrate
    structure.name = 'thin_layer_sample_1'
    return Sample(structure)

def thin_layer_sample_2():
    """Defines a 3-layer sample with thin layers.

    Returns:
        structures.Sample: structure in format for information calculation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(4, name='Layer 1')(thick=200, rough=2)
    layer2 = refnx.reflect.SLD(5, name='Layer 2')(thick=30, rough=6)
    layer3 = refnx.reflect.SLD(6, name='Layer 3')(thick=6, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | layer3 | substrate
    structure.name = 'thin_layer_sample_2'
    return Sample(structure)

def similar_sld_sample_1():
    """Defines a 2-layer sample with layers of similar SLD.

    Returns:
        structures.Sample: structure in format for information calculation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(0.9, name='Layer 1')(thick=80, rough=2)
    layer2 = refnx.reflect.SLD(1.0, name='Layer 2')(thick=50, rough=6)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | substrate
    structure.name = 'similar_sld_sample_1'
    return Sample(structure)

def similar_sld_sample_2():
    """Defines a 3-layer sample with layers of similar SLD.

    Returns:
        structures.Sample: structure in format for information calculation.

    """
    air = refnx.reflect.SLD(0, name='Air')
    layer1 = refnx.reflect.SLD(3.0, name='Layer 1')(thick=50, rough=2)
    layer2 = refnx.reflect.SLD(5.5, name='Layer 2')(thick=30, rough=6)
    layer3 = refnx.reflect.SLD(6.0, name='Layer 3')(thick=35, rough=2)
    substrate = refnx.reflect.SLD(2.047, name='Substrate')(thick=0, rough=2)

    structure = air | layer1 | layer2 | layer3 | substrate
    structure.name = 'similar_sld_sample_2'
    return Sample(structure)

def refnx_to_refl1d(sample):
    """Converts a standard refnx structure to an equivalent Refl1D structure.

    Args:
        sample (refnx.reflect.Structure): refnx structure to convert.

    Returns:
        refl1d.model.Stack: equivalent structure defined in Refl1D.

    """
    # Iterate over each component.
    structure = refl1d.material.SLD(rho=0, name='Air')
    for component in sample[1:]:
        name, sld = component.name, component.sld.real.value,
        thick, rough = component.thick.value, component.rough.value

        # Add the component in the opposite direction to the refnx definition.
        structure = refl1d.material.SLD(rho=sld, name=name)(thick, rough) | structure

    structure.name = sample.name
    return structure

if __name__ == '__main__':

    structures = [simple_sample, many_param_sample,
                  thin_layer_sample_1, thin_layer_sample_2,
                  similar_sld_sample_1, similar_sld_sample_2,
                  YIG, SymmetricBilayer, SingleAsymmetricBilayer]

    save_path = './results'

    # Plot the SLD and reflectivity profiles of all structures in this file.
    for structure in structures:
        sample = structure()
        
        sample.sld_profile(save_path)
        plt.close()
        
        sample.reflectivity_profile(save_path)
        plt.close()
