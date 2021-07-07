import numpy as np
import os, sys, time
sys.path.append('./')

from scipy.optimize import differential_evolution, NonlinearConstraint
from structures import VariableAngle, VariableContrast

class Optimiser:
    """Contains code for optimising a neutron reflectometry experiment.

    Attributes:
        sample (structures.Sample): sample to optimise an experiment for.

    """
    def __init__(self, sample):
        self.sample = sample

    def optimise_angle_times(self, num_angles, contrasts=[], total_time=1000, points=100,
                             angle_bounds=(0.2, 2.3), workers=-1, verbose=True):
        """Optimises the measurement angles and associated counting times for an experiment,
           given a fixed time budget.

        Args:
            num_angles (int): number of angles to optimise.
            contrasts (list): contrasts of the experiment, if applicable.
            total_time (float): time budget for experiment.
            points (int): number of data points to use for each angle.
            angle_bounds (tuple): interval containing angles to consider.
            workers (int): number of CPU cores to use when optimising (-1 is all available).
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised angles, optimised counting times and optimisation function value.

        """
        # Check that the measurement angle of the sample can be varied.
        assert isinstance(self.sample, VariableAngle)

        # Define the bounds of each condition to optimise.
        bounds = [angle_bounds]*num_angles + [(0, total_time)]*num_angles
        # Arguments for optimisation function
        args = [num_angles, contrasts, points]

        # Constrain the times to sum to the fixed time budget.
        constraints = [NonlinearConstraint(lambda x: sum(x[num_angles:]), total_time, total_time)]

        # Optimise angles and times, and return results.
        res, val = Optimiser.__optimise(self._angle_times_func, bounds, constraints, args, workers, verbose)
        return res[:num_angles], res[num_angles:], val

    def optimise_contrasts(self, num_contrasts, angle_times, contrast_bounds=(-0.56, 6.36),
                           workers=-1, verbose=True):
        """Finds the optimal contrast SLDs for an experiment.

        Args:
            num_contrasts (int): number of contrasts to optimise.
            angle_times (list): number of points and counting times for each angle to simulate.
            contrast_bounds (tuple): interval containing contrast SLDs to consider.
            workers (int): number of CPU cores to use when optimising (-1 is all available).
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised contrast SLDs and optimisation function value.

        """
        # Check that the contrast SLD for the sample can be varied.
        assert isinstance(self.sample, VariableContrast)

        # Define the bounds of each condition to optimise.
        bounds = [contrast_bounds]*num_contrasts
        # Arguments for optimisation function
        args = [num_contrasts, angle_times]

        # Optimise contrasts and return results.
        return Optimiser.__optimise(self._contrasts_func, bounds, [], args, workers, verbose)

    def _angle_times_func(self, x, num_angles, contrasts, points):
        """Defines the optimisation function for optimising an experiment's measurement
           angles and associated counting times.

        Args:
            x (list): conditions to calculate optimisation function with.
            num_angles (int): number of angles being optimised.
            contrasts (list): contrasts of the experiment, if applicable.
            points (int): number of data points to use for each angle.

        Returns:
            float: negative of minimum eigenvalue of Fisher information matrix using given conditions.

        """
        # Extract the angles and counting times from given parameter array.
        angle_times = [(x[i], points, x[num_angles+i]) for i in range(num_angles)]

        # Calculate the Fisher information matrix.
        g = self.sample.angle_info(angle_times, contrasts)

        # Return negative of minimum eigenvalue as optimisation algorithm is minimising.
        return -np.linalg.eigvalsh(g)[0]

    def _contrasts_func(self, x, num_contrasts, angle_times):
        """Defines the optimisation function for optimising an experiment's contrasts.

        Args:
            x (type): conditions to calculate optimisation function with.
            num_contrasts (type): number of contrasts being optimised.
            angle_times (type): number of points and counting times for each angle to simulate.

        Returns:
            float: negative of minimum eigenvalue of Fisher information matrix using given conditions.

        """
        # Calculate the Fisher information matrix.
        g = self.sample.contrast_info(angle_times, x)

        # Return negative of minimum eigenvalue as optimisation algorithm is minimising.
        return -np.linalg.eigvalsh(g)[0]

    @staticmethod
    def __optimise(func, bounds, constraints, args, workers, verbose):
        """Optimises a given `func` using the differential evolution global optimisation algorithm.

        Args:
            func (callable): function to optimise.
            bounds (list): permissible values for the conditions to optimise.
            constraints (list): constraints on conditions to optimise.
            args (list): arguments for optimisation function.
            workers (int): number of CPU cores to use when optimising (-1 is all available).
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised experimental conditions and optimisation function value.

        """
        # Run differential evolution on the given optimisation function.
        res = differential_evolution(func, bounds, constraints=constraints,
                                    args=args, polish=False, tol=0.001,
                                    updating='deferred', workers=workers,
                                    disp=verbose)
        return res.x, res.fun

def _angle_results(optimiser, contrasts, total_time, save_path='./results'):
    """Optimises the measurement angles and associated counting times for an experiment
       using different numbers of angles.

    Args:
        optimiser (optimise.Optimiser): optimiser for the experiment.
        contrasts (list): contrasts of the experiment, if applicable.
        total_time (float): total time budget for the experiment.
        save_path (str): path to directory to save results to.

    """
    save_path = os.path.join(save_path, optimiser.sample.name)

    # Create a new text file for the results.
    with open(os.path.join(save_path, 'optimised_angles.txt'), 'w') as file:
        # Optimise the experiment using 1-4 angles.
        for i, num_angles in enumerate([1, 2, 3, 4]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 4))

            # Time how long the optimisation takes.
            start = time.time()
            angles, times, val = optimiser.optimise_angle_times(num_angles, contrasts, total_time, verbose=False)
            end = time.time()

            # Round the optimisation function value to 4 significant figures.
            val = np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')

            # Write the optimised conditions, objective value and computation time to the results file.
            file.write('----------- {} Angles -----------\n'.format(num_angles))
            file.write('Angles: {}\n'.format(list(np.round(angles, 2))))
            file.write('Times: {}\n'.format(list(np.round(times, 1))))
            file.write('Objective value: {}\n'.format(val))
            file.write('Computation time: {}\n\n'.format(round(end-start, 1)))

def _contrast_results(optimiser, angle_times, save_path='./results'):
    """Optimises the contrasts for an experiment using different numbers of contrasts.

    Args:
        optimiser (optimise.Optimiser): optimiser for the experiment.
        angle_times (list): number of points and counting times for each angle to simulate.
        save_path (str): path to directory to save results to.

    """
    save_path = os.path.join(save_path, optimiser.sample.name)

    # Create a new text file for the results.
    with open(os.path.join(save_path, 'optimised_contrasts.txt'), 'w') as file:
        # Optimise the experiment using 1-4 contrasts.
        for i, num_contrasts in enumerate([1, 2, 3, 4]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 4))

            # Split the measurement times equally over each contrast.
            new_angle_times = [(angle, points, time/num_contrasts) for angle, points, time in angle_times]

            # Time how long the optimisation takes.
            start = time.time()
            contrasts, val = optimiser.optimise_contrasts(num_contrasts, new_angle_times, verbose=False)
            end = time.time()

            # Round the optimisation function value to 4 significant figures.
            val = np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')

            # Write the optimised conditions, objective value and computation time to the results file.
            file.write('----------- {} Contrasts -----------\n'.format(num_contrasts))
            file.write('Contrasts: {}\n'.format(list(np.round(contrasts, 2))))
            file.write('Objective value: {}\n'.format(val))
            file.write('Computation time: {}\n\n'.format(round(end-start, 1)))

if __name__ == '__main__':
    from structures import SymmetricBilayer, SingleAsymmetricBilayer

    sample = SingleAsymmetricBilayer()
    optimiser = Optimiser(sample)

    contrasts = [6.36]
    total_time = 1000
    _angle_results(optimiser, contrasts, total_time)

    angle_times = [(0.5, 150, 55), (2.3, 150, 945)]
    _contrast_results(optimiser, angle_times)
