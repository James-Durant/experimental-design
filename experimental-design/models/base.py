import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.join(__file__, '..'))
plt.rcParams['figure.figsize'] = (9,7)
plt.rcParams['figure.dpi'] = 600

from abc import ABC, abstractmethod

import refnx.dataset, refnx.reflect, refnx.analysis

from simulate import simulate
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
    """Abstract class representing a neutron reflectometry sample."""
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

class BaseLipid(BaseSample, VariableContrast, VariableUnderlayer):
    """Abstract class representing the base class for a lipid model."""
    def __init__(self):
        # Load experimentally-measured data.
        self._create_objectives()

    @abstractmethod
    def _create_objectives(self):
        """Loads the measured data for the lipid sample."""
        pass

    def angle_info(self, angle_times, contrasts):
        """Calculates the Fisher information matrix for the lipid sample measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, None)

    def contrast_info(self, angle_times, contrasts):
        """Calculates the Fisher information matrix for the lipid sample with contrasts
           measured over a number of angles.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        return self.__conditions_info(angle_times, contrasts, None)

    def underlayer_info(self, angle_times, contrasts, underlayers):
        """Calculates the Fisher information matrix for the lipid sample with `underlayers`,
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
        """Calculates the Fisher information matrix for the lipid sample with given conditions.

        Args:
            angle_times (list): points and counting times for each measurement angle to simulate.
            contrasts (list): SLDs of contrasts to simulate.
            underlayers (list): thickness and SLD of each underlayer to add.

        Returns:
            numpy.ndarray: Fisher information matrix.

        """
        qs, counts, models = [], [], []
        for contrast in contrasts:
            model, data = simulate(self._using_conditions(contrast, underlayers),
                                   angle_times, scale=1, bkg=5e-6, dq=2)
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

    def sld_profile(self, save_path, filename='sld_profile', ylim=None, legend=True):
        """Plots the SLD profile of the lipid sample.

        Args:
            save_path (str): path to directory to save SLD profile to.
            filename
            ylim
            legend

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Plot the SLD profile for each measured contrast.
        for structure in self.structures:
            ax.plot(*structure.sld_profile(self.distances))

        ax.set_xlabel('$\mathregular{Distance\ (\AA)}$', fontsize=11, weight='bold')
        ax.set_ylabel('$\mathregular{SLD\ (10^{-6} \AA^{-2})}$', fontsize=11, weight='bold')
        
        if ylim:
            ax.set_ylim(*ylim)
            
        if legend:
            ax.legend(self.labels)

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
        ax.set_xscale('log')
        ax.set_yscale('log')
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
