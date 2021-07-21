import matplotlib.pyplot as plt
import numpy as np
import os
#plt.rcParams['figure.figsize'] = (4.5,7)
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

    def reflectivity_profile(self, save_path, q_min=0.005, q_max=0.3, points=500, scale=1, bkg=0, dq=2):
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

    def to_refl1d(self):
        """Converts a standard refnx structure to an equivalent Refl1D structure.
    
        Args:
            sample (refnx.reflect.Structure): refnx structure to convert.
    
        Returns:
            refl1d.model.Stack: equivalent structure defined in Refl1D.
    
        """
        # Iterate over each component.
        structure = refl1d.material.SLD(rho=0, name='Air')
        for component in self.structure[1:]:
            name, sld = component.name, component.sld.real.value,
            thick, rough = component.thick.value, component.rough.value
    
            # Add the component in the opposite direction to the refnx definition.
            structure = refl1d.material.SLD(rho=sld, name=name)(thick, rough) | structure
    
        structure.name = self.structure.name
        self.structure = structure
    
    def to_refnx(self):
        # Iterate over each component.
        structure = refnx.reflect.SLD(0, name='Air')
        for component in list(reversed(self.structure))[1:]:
            name, sld = component.name, component.material.rho.value,
            thick, rough = component.thickness.value, component.interface.value
            
            structure |= refnx.reflect.SLD(sld, name=name)(thick, rough)
    
        structure.name = self.structure.name
        self.structure = structure

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
        
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        count = 0
        for probe, qr in zip(self.experiment.probe.xs, self.experiment.reflectivity()):
            if qr is not None:
                ax.errorbar(probe.Q, probe.R, probe.dR, marker='o', ms=2, lw=0, elinewidth=0.5, capsize=0.5, label=self.labels[count]+' Data', color=colours[count])
                ax.plot(probe.Q, qr[1], color=colours[count], zorder=20, label=self.labels[count]+' Fitted')
                count += 1
    
        ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
        ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
        ax.set_yscale('log')
        ax.legend()

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'reflectivity_profile')

class YIG_Sample(MagneticSample, VariableUnderlayer):
    def __init__(self):
        self.name = 'YIG_sample'
        
        self.data_path = '../experimental-design/data/YIG_sample'
        self.labels = ['Up', 'Down']
        self.scale = 1.025
        self.bkg = 4e-7
        self.dq = 2.8
        self.mag_angle = 90

        self.Pt_sld = bumps.parameter.Parameter(5.646, name='Pt SLD')
        self.Pt_thick = bumps.parameter.Parameter(21.08, name='Pt Thickness')
        self.Pt_rough = bumps.parameter.Parameter(8.211, name='Air|Pt Roughness')
        self.Pt_mag = bumps.parameter.Parameter(0.0128, name='Pt Magnetic SLD')
        
        self.FePt_sld = bumps.parameter.Parameter(4.678, name='FePt SLD')
        self.FePt_thick = bumps.parameter.Parameter(19.67, name='FePt Thickness')
        self.FePt_rough = bumps.parameter.Parameter(2, name='Pt|FePt Roughness')
        
        self.YIG_sld = bumps.parameter.Parameter(5.836, name='YIG SLD')
        self.YIG_thick = bumps.parameter.Parameter(713.8, name='YIG Thickness')
        self.YIG_rough = bumps.parameter.Parameter(13.55, name='FePt|YIG Roughness')
        self.YIG_mag = bumps.parameter.Parameter(0.349, name='YIG Magnetic SLD')
        
        self.sub_sld = bumps.parameter.Parameter(5.304, name='Substrate SLD')
        self.sub_rough = bumps.parameter.Parameter(30, name='YIG|Substrate Roughness')
                
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
        
        self.Pt_sld.range(5, 6)
        self.Pt_thick.range(2, 30)
        self.Pt_rough.range(0, 9)
        self.Pt_mag.range(0, 0.2)
        
        self.FePt_sld.range(4.5, 5.5)
        self.FePt_thick.range(0, 25)
        self.FePt_rough.range(2, 10)
        
        self.YIG_sld.range(5, 6)
        self.YIG_thick.range(100, 900)
        self.YIG_rough.range(0, 70)
        self.YIG_mag.range(0, 0.6)
        
        self.sub_sld.range(4.5, 5.5)
        self.sub_rough.range(20, 30)
        
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
    
    def angle_info(self, angle_times, contrasts=None):
        models, datasets = simulate_magnetic(self.structure, angle_times, bkg=self.bkg, dq=self.dq,
                                             pp=True, pm=False, mp=False, mm=True)

        qs = [data[:,0] for data in datasets]
        counts = [data[:,3] for data in datasets]

        return fisher(qs, self.params, counts, models)
    
    def underlayer_info(self):
        pass
    
    def nested_sampling(self, angle_times, save_path, filename, dynamic=False):
        """Runs nested sampling on simulated data of the sample.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            save_path (str): path to directory to save corner plot to.
            filename (str): name of file to save corner plot to.
            dynamic (bool): whether to use static or dynamic nested sampling.

        """
        #models, _ = simulate_magnetic(self.structure, angle_times, bkg=self.bkg, dq=self.dq,
        #                              pp=True, pm=False, mp=False, mm=True)
        #xs = [models[0].probe.xs[0], None, None, models[1].probe.xs[3]]
        
        #experiment = refl1d.experiment.Experiment(sample=self.structure, probe=refl1d.probe.PolarizedQProbe(xs=xs, name=''))
        objective = bumps.fitproblem.FitProblem(self.experiment)
  
        # Sample the objective using nested sampling.
        sampler = Sampler(objective)
        fig = sampler.sample(dynamic=dynamic)

        # Save the sampling corner plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, 'nested_sampling_'+filename)

class BaseLipid(BaseSample, VariableContrast, VariableUnderlayer):
    """Abstract class representing the base class for a lipid model."""
    def __init__(self):
        self._create_objectives()

    @abstractmethod
    def _create_objectives(self):
        """Loads the measured data for the lipid sample."""
        pass

    def angle_info(self, angle_times, contrasts):
        """Calculates the Fisher information matrix for a lipid sample measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, None)

    def contrast_info(self, angle_times, contrasts):
        """Calculates the Fisher information matrix for a lipid sample with contrasts
           measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, None)

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
        return self.__conditions_info(angle_times, contrasts, underlayers)

    def __conditions_info(self, angle_times, contrasts, underlayers):
        """Calculates the Fisher information matrix for a lipid sample with given conditions.

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

    def sld_profile(self, save_path, filename='sld_profile'):
        """Plots the SLD profile of the lipid sample.

        Args:
            save_path (str): path to directory to save SLD profile to.
            filename

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the SLD profile for each measured contrast.
        for structure in self.structures:
            ax.plot(*structure.sld_profile(self.distances))

        ax.set_xlabel('$\mathregular{Distance\ (\AA)}$', fontsize=11, weight='bold')
        ax.set_ylabel('$\mathregular{SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
        ax.legend(self.labels)
        #ax.set_ylim(-0.6, 7.5)

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, filename)

    def reflectivity_profile(self, save_path, filename='reflectivity_profile'):
        """Plots the reflectivity profile of the lipid sample.

        Args:
            save_path (str): path to directory to save reflectivity profile to.
            filename

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Iterate over each measured contrast.
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
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
            ax.errorbar(q, r, dr, marker='o', ms=3, lw=0, elinewidth=1, capsize=1.5,
                        color=colours[i], label=label)
            ax.plot(q, r_model, color=colours[i], zorder=20)

        ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
        ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
        #ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(0.005, 0.3)
        ax.set_ylim(1e-10, 3)
        ax.legend()

        # Save the plot.
        save_path = os.path.join(save_path, self.name)
        save_plot(fig, save_path, filename)

    def nested_sampling(self, contrasts, angle_times, save_path, filename, underlayers=None, dynamic=False):
        """Runs nested sampling on simulated data of the lipid sample.

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
        save_plot(fig, save_path, 'nested_sampling_'+filename)

class Monolayer(BaseLipid):
    def __init__(self, deuterated=False):
        self.name = 'monolayer'
        self.data_path = '../experimental-design/data/monolayer'
        self.labels = ['Hydrogenated-D2O', 'Deuterated-NRW', 'Hydrogenated-NRW']
        self.distances = np.linspace(-25, 90, 500)
        
        self.deuterated = deuterated
        
        self.scales = [1.8899, 1.8832, 1.8574]
        self.bkgs = [3.565e-6, 5.348e-6, 6.542e-6]
        self.dq = 3

        self.air_tg_rough    = refnx.analysis.Parameter( 5.0000, 'Air|Tailgroup Roughness',   ( 5, 8))
        self.lipid_apm       = refnx.analysis.Parameter(54.1039, 'Lipid Area Per Molecule',   (30, 80))
        self.hg_waters       = refnx.analysis.Parameter( 6.6874, 'Headgroup Bound Waters',    ( 0, 20))
        self.monolayer_rough = refnx.analysis.Parameter( 2.0233, 'Monolayer Roughness',       ( 0, 10))
        self.non_lipid_vf    = refnx.analysis.Parameter( 0.0954, 'Non-lipid Volume Fraction', ( 0, 1))
        self.protein_tg      = refnx.analysis.Parameter( 0.9999, 'Protein Tails',             ( 0, 1))
        self.protein_hg      = refnx.analysis.Parameter( 1.0000, 'Protein Headgroup',         ( 0, 1))
        self.protein_thick   = refnx.analysis.Parameter(32.6858, 'Protein Thickness',         ( 0, 80))
        self.protein_vfsolv  = refnx.analysis.Parameter( 0.5501, 'Protein Hydration',         ( 0, 100))
        self.water_rough     = refnx.analysis.Parameter( 3.4590, 'Protein|Water Roughness',   ( 0, 15))
        
        # Exclude protein parameters
        self.params = [self.lipid_apm]
        
        # Vary all of the parameters defined above.
        for param in self.params:
            param.vary=True
        
        super().__init__()
        
    def _using_conditions(self, contrast_sld, underlayers=None, deuterated=None, protein=False):
        assert underlayers is None
        
        if deuterated is None:
            deuterated = self.deuterated
        
        contrast_sld *= 1e-6
        
        # Define known SLDs of D2O and H2O
        d2o_sld =  6.35e-6
        h2o_sld = -0.56e-6
        
        # Define known protein SLDs for D2O and H2O.
        protein_d2o_sld = 3.4e-6
        protein_h2o_sld = 1.9e-6
        
        # Define neutron scattering lengths.
        carbon_sl     =  0.6646e-4
        oxygen_sl     =  0.5843e-4
        hydrogen_sl   = -0.3739e-4
        phosphorus_sl =  0.5130e-4
        deuterium_sl  =  0.6671e-4
        
        # Calculate the total scattering length in each fragment.
        COO  = 1*carbon_sl     + 2*oxygen_sl
        GLYC = 3*carbon_sl     + 5*hydrogen_sl
        CH3  = 1*carbon_sl     + 3*hydrogen_sl        
        PO4  = 1*phosphorus_sl + 4*oxygen_sl
        CH2  = 1*carbon_sl     + 2*hydrogen_sl
        H2O  = 2*hydrogen_sl   + 1*oxygen_sl
        D2O  = 2*deuterium_sl  + 1*oxygen_sl
        CD3  = 1*carbon_sl     + 3*deuterium_sl       
        CD2  = 1*carbon_sl     + 2*deuterium_sl
        
        # Volumes of each fragment.
        vCH3  = 52.7/2
        vCH2  = 28.1
        vCOO  = 39.0
        vGLYC = 68.8
        vPO4  = 53.7
        vWAT  = 30.4
        
        # Calculate volumes from components.
        hg_vol = vPO4 + 2*vGLYC + 2* vCOO
        tg_vol = 28*vCH2 + 2*vCH3
        
        # Calculate mole fraction of D2O from the bulk SLD.
        d2o_molfr = (contrast_sld - h2o_sld) / (d2o_sld - h2o_sld)
        
        # Calculate 'average' scattering length sum per water molecule in bulk.
        sl_sum_water = d2o_molfr*D2O + (1-d2o_molfr)*H2O
        
        # Calculate scattering length sums for the other fragments.
        sl_sum_hg = PO4 + 2*GLYC + 2*COO
        sl_sum_tg_h = 28*CH2 + 2*CH3
        sl_sum_tg_d = 28*CD2 + 2*CD3
        
        # Need to include the number of hydrating water molecules in headgroup
        # scattering length sum and headgroup volume.
        lipid_total_sl_sum = sl_sum_water * self.hg_waters
        lipid_total_vol = vWAT * self.hg_waters
        
        hg_vol = hg_vol + lipid_total_vol
        sl_sum_hg = sl_sum_hg + lipid_total_sl_sum
        
        hg_sld = sl_sum_hg / hg_vol
        
        # Calculate the SLD of the hydrogenated and deuterated tailgroups.
        tg_h_sld = sl_sum_tg_h / tg_vol
        tg_d_sld = sl_sum_tg_d / tg_vol
        
        if protein:
            # Contrast_point calculation.
            contrast_point = (contrast_sld - h2o_sld) / (d2o_sld - h2o_sld)
            
            # Calculated SLD of protein and hydration
            protein_sld = (contrast_point * protein_d2o_sld) + ((1-contrast_point) * protein_h2o_sld)
            
            # Bulk in is 0 SLD so no extra terms.
            protein_tg_sld = self.protein_tg * protein_sld
            protein_hg_sld = self.protein_hg * protein_sld
            
            hg_sld = (1-self.non_lipid_vf)*hg_sld + self.non_lipid_vf*protein_hg_sld
            
            tg_h_sld = (1-self.non_lipid_vf)*tg_h_sld + self.non_lipid_vf*protein_tg_sld
            tg_d_sld = (1-self.non_lipid_vf)*tg_d_sld + self.non_lipid_vf*protein_tg_sld
        
            protein = refnx.reflect.SLD(protein_sld*1e6, name='Protein')(self.protein_thick, self.monolayer_rough, self.protein_vfsolv)
        
        # Tailgroup and headgroup thicknesses.
        tg_thick = tg_vol / self.lipid_apm
        hg_thick = hg_vol / self.lipid_apm
        
        # Define the structure.
        air = refnx.reflect.SLD(0, name='Air')
        tg_h = refnx.reflect.SLD(tg_h_sld*1e6, name='Hydrogenated Tailgroup')(tg_thick, self.air_tg_rough)
        tg_d = refnx.reflect.SLD(tg_d_sld*1e6, name='Deuterated Tailgroup')(tg_thick, self.air_tg_rough)
        hg = refnx.reflect.SLD(hg_sld*1e6, name='Headgroup')(hg_thick, self.monolayer_rough)
        water = refnx.reflect.SLD(contrast_sld*1e6, name='Water')(0, self.water_rough)
        
        structure = air
        if deuterated:
            structure |= tg_d
        else:
            structure |= tg_h
            
        if protein:
            return structure | hg | protein | water
        else:
            return structure | hg | water

    def _create_objectives(self, protein=True):
        nrw, d2o = 0.1, 6.35
    
        self.structures = [self._using_conditions(d2o, deuterated=False, protein=protein),
                           self._using_conditions(nrw, deuterated=True,  protein=protein),
                           self._using_conditions(nrw, deuterated=False, protein=protein)]
        
        models = [refnx.reflect.ReflectModel(structure, scale=scale, bkg=bkg*scale, dq=self.dq)
                  for structure, scale, bkg in zip(self.structures, self.scales, self.bkgs)]
        
        datasets = [refnx.dataset.ReflectDataset(os.path.join(self.data_path, '{}.dat'.format(label)))
                    for label in self.labels]
    
        self.objectives = [refnx.analysis.Objective(model, data)
                           for model, data in zip(models, datasets)]

    def sld_profile(self, save_path):
        self._create_objectives(protein=False)
        super().sld_profile(save_path, 'sld_profile_no_protein')
        self._create_objectives(protein=True)
        super().sld_profile(save_path, 'sld_profile_protein')

class SymmetricBilayer(BaseLipid):
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
        self.name = 'symmetric_bilayer'
        self.data_path = '../experimental-design/data/symmetric_bilayer'
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

class AsymmetricBilayer(BaseLipid):
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
            sio2.vfsolv.value = 0
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

if __name__ == '__main__':
    structures = [simple_sample, many_param_sample,
                  thin_layer_sample_1, thin_layer_sample_2,
                  similar_sld_sample_1, similar_sld_sample_2,
                  YIG_Sample, Monolayer,
                  SymmetricBilayer, SingleAsymmetricBilayer]

    save_path = './results'

    # Plot the SLD and reflectivity profiles of all structures in this file.
    for structure in structures:
        sample = structure()
        
        sample.sld_profile(save_path)    
        sample.reflectivity_profile(save_path)
        
        plt.close('all')
