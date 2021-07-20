import numpy as np
import os, sys, time
sys.path.append('./')

from scipy.optimize import differential_evolution, NonlinearConstraint
from structures import VariableAngle, VariableContrast, VariableUnderlayer

class Optimiser:
    """Contains code for optimising a neutron reflectometry experiment.

    Attributes:
        sample (structures.Sample): sample to optimise an experiment for.

    """
    def __init__(self, sample):
        self.sample = sample

    def optimise_angle_times(self, num_angles, contrasts=[], total_time=1000,
                             angle_bounds=(0.2, 4), points=100, workers=-1, verbose=True):
        """Optimises the measurement angles and associated counting times for an experiment,
           given a fixed time budget.

        Args:
            num_angles (int): number of angles to optimise.
            contrasts (list): contrasts of the experiment, if applicable.
            total_time (float): time budget for experiment.
            angle_bounds (tuple): interval containing angles to consider.
            points (int): number of data points to use for each angle.
            workers (int): number of CPU cores to use when optimising (-1 is all available).
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised angles, optimised counting times and optimisation function value.

        """
        # Check that the measurement angle of the sample can be varied.
        assert isinstance(self.sample, VariableAngle)

        # Define the bounds of each condition to optimise (angles and time splits).
        bounds = [angle_bounds]*num_angles + [(0, 1)]*num_angles

        # Arguments for optimisation function
        args = [num_angles, contrasts, total_time, points]

        # Constrain the counting times to sum to the fixed time budget.
        constraints = [NonlinearConstraint(lambda x: sum(x[num_angles:]), 1, 1)]

        # Optimise angles and times, and return results.
        res, val = Optimiser.__optimise(self._angle_times_func, bounds, constraints, args, workers, verbose)
        return res[:num_angles], res[num_angles:], val

    def optimise_contrasts(self, num_contrasts, angle_splits, total_time=1000,
                           contrast_bounds=(-0.56, 6.36), workers=-1, verbose=True):
        """Finds the optimal contrast SLDs for an experiment, given a fixed time budget.

        Args:
            num_contrasts (int): number of contrasts to optimise.
            angle_splits (list): points and proportion of total counting time for each angle.
            total_time (float): time budget for experiment.
            contrast_bounds (tuple): interval containing contrast SLDs to consider.
            workers (int): number of CPU cores to use when optimising (-1 is all available).
            verbose (bool): whether to display progress or not.

        Returns:
            tuple: optimised contrast SLDs, counting time proportions and optimisation function value.

        """
        # Check that the contrast SLD for the sample can be varied.
        assert isinstance(self.sample, VariableContrast)

        # Define the bounds of each condition to optimise (contrast SLDs and time splits).
        bounds = [contrast_bounds]*num_contrasts + [(0, 1)]*num_contrasts

        # Constrain the counting times of each contrast to sum to the fixed time budget.
        constraints = [NonlinearConstraint(lambda x: sum(x[num_contrasts:]), 1, 1)]

        # Arguments for optimisation function.
        args = [num_contrasts, angle_splits, total_time]

        # Optimise contrasts, counting time splits and return the results.
        res, val = Optimiser.__optimise(self._contrasts_func, bounds, constraints, args, workers, verbose)
        return res[:num_contrasts], res[num_contrasts:], val

    def optimise_underlayers(self, num_underlayers, angle_times, contrasts, 
                             thick_bounds=(0, 500), sld_bounds=(1, 9),
                             workers=-1, verbose=True):

        # Check that the underlayers of the sample can be varied.
        assert isinstance(self.sample, VariableUnderlayer)

        # Define the bounds of each condition to optimise (underlayer thicknesses and SLDs).
        bounds = [thick_bounds]*num_underlayers + [sld_bounds]*num_underlayers

        # Arguments for optimisation function.
        args = [num_underlayers, angle_times, contrasts]

        # Optimise contrasts, counting time splits and return the results.
        res, val = Optimiser.__optimise(self._underlayers_func, bounds, [], args, workers, verbose)
        return res[:num_underlayers], res[num_underlayers:], val

    def _angle_times_func(self, x, num_angles, contrasts, total_time, points):
        """Defines the optimisation function for optimising an experiment's measurement
           angles and associated counting times.

        Args:
            x (list): conditions to calculate optimisation function with.
            num_angles (int): number of angles being optimised.
            contrasts (list): contrasts of the experiment, if applicable.
            total_time (float): time budget for experiment.
            points (int): number of data points to use for each angle.

        Returns:
            float: negative of minimum eigenvalue of Fisher information matrix using given conditions.

        """
        # Extract the angles and counting times from given parameter array.
        angle_times = [(x[i], points, total_time*x[num_angles+i]) for i in range(num_angles)]

        # Calculate the Fisher information matrix.
        g = self.sample.angle_info(angle_times, contrasts)

        # Return negative of minimum eigenvalue as optimisation algorithm is minimising.
        return -np.linalg.eigvalsh(g)[0]

    def _contrasts_func(self, x, num_contrasts, angle_splits, total_time):
        """Defines the optimisation function for optimising an experiment's contrasts.

        Args:
            x (type): conditions to calculate optimisation function with.
            num_contrasts (int): number of contrasts to optimise.
            angle_splits (type): points and proportion of total counting time for each angle.
            total_time (float): time budget for experiment.

        Returns:
            float: negative of minimum eigenvalue of Fisher information matrix using given conditions.

        """
        # Calculate the Fisher information matrix.
        m = len(self.sample.params)
        g = np.zeros((m, m))
        # Iterate over each contrast.
        for i in range(num_contrasts):
            # Calculate the proportion of the total counting time for each angle.
            angle_times = [(angle, points, total_time*x[num_contrasts+i]*split) for angle, points, split in angle_splits]
            g += self.sample.contrast_info(angle_times, [x[i]])

        # Return negative of minimum eigenvalue as optimisation algorithm is minimising.
        return -np.linalg.eigvalsh(g)[0]
    
    def _underlayers_func(self, x, num_underlayers, angle_times, contrasts):
        underlayers = [(x[i], x[num_underlayers+i]) for i in range(num_underlayers)]
        g = self.sample.underlayer_info(angle_times, contrasts, underlayers)
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

def _angle_results(optimiser, total_time, angle_bounds, save_path='./results'):
    """Optimises the measurement angles and associated counting times for an experiment
       using different numbers of angles.

    Args:
        optimiser (optimise.Optimiser): optimiser for the experiment.
        total_time (float): total time budget for the experiment.
        angle_bounds (tuple): interval containing angles to consider.
        save_path (str): path to directory to save results to.

    """
    save_path = os.path.join(save_path, optimiser.sample.name)

    for contrast, filename in [(6.36, 'D2O'), (2.07, 'SMW'), (-0.56, 'H2O')]:
        print(filename)
        # Create a new text file for the results.
        with open(os.path.join(save_path, 'optimised_angles_{}.txt'.format(filename)), 'w') as file:
            # Optimise the experiment using 1-4 angles.
            for i, num_angles in enumerate([1, 2, 3, 4]):
                # Display progress.
                print('>>> {0}/{1}'.format(i, 4))

                # Time how long the optimisation takes.
                start = time.time()
                angles, splits, val = optimiser.optimise_angle_times(num_angles, [contrast], total_time, angle_bounds, verbose=False)
                end = time.time()

                # Convert to percentages.
                splits = np.array(splits)*100

                # Round the optimisation function value to 4 significant figures.
                val = np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')

                # Write the optimised conditions, objective value and computation time to the results file.
                file.write('----------- {} Angles -----------\n'.format(num_angles))
                file.write('Angles: {}\n'.format(list(np.round(angles, 2))))
                file.write('Splits (%): {}\n'.format(list(np.round(splits, 1))))
                file.write('Objective value: {}\n'.format(val))
                file.write('Computation time: {}\n\n'.format(round(end-start, 1)))

        print()

def _contrast_results(optimiser, total_time, angle_splits, contrast_bounds, save_path='./results'):
    """Optimises the contrasts of an experiment using different numbers of contrasts.

    Args:
        optimiser (optimise.Optimiser): optimiser for the experiment.
        total_time (float): time budget for experiment.
        angle_splits (list): points and proportion of total counting time for each angle.
        contrast_bounds (tuple): interval containing contrast SLDs to consider.

        save_path (str): path to directory to save results to.

    """
    save_path = os.path.join(save_path, optimiser.sample.name)

    # Create a new text file for the results.
    with open(os.path.join(save_path, 'optimised_contrasts.txt'), 'w') as file:
        # Optimise the experiment using 1-4 contrasts.
        for i, num_contrasts in enumerate([1, 2, 3, 4]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 4))

            # Time how long the optimisation takes.
            start = time.time()
            contrasts, splits, val = optimiser.optimise_contrasts(num_contrasts, angle_splits, total_time, contrast_bounds, verbose=False)
            end = time.time()
            
            # Convert to percentages.
            splits = np.array(splits)*100

            # Round the optimisation function value to 4 significant figures.
            val = np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')

            # Write the optimised conditions, objective value and computation time to the results file.
            file.write('----------- {} Contrasts -----------\n'.format(num_contrasts))
            file.write('Contrasts: {}\n'.format(list(np.round(contrasts, 2))))
            file.write('Splits (%): {}\n'.format(list(np.round(splits, 1))))
            file.write('Objective value: {}\n'.format(val))
            file.write('Computation time: {}\n\n'.format(round(end-start, 1)))

def _underlayer_results(optimiser, angle_times, contrasts, thick_bounds, sld_bounds, save_path='./results'):
    save_path = os.path.join(save_path, optimiser.sample.name)

    # Create a new text file for the results.
    with open(os.path.join(save_path, 'optimised_underlayers.txt'), 'w') as file:
        # Optimise the experiment using 1-4 contrasts.
        for i, num_underlayers in enumerate([1, 2, 3]):
            # Display progress.
            print('>>> {0}/{1}'.format(i, 3))

            # Time how long the optimisation takes.
            start = time.time()
            thicknesses, slds, val = optimiser.optimise_underlayers(num_underlayers, angle_times, contrasts, 
                                                                    thick_bounds, sld_bounds, verbose=False)
            end = time.time()

            # Round the optimisation function value to 4 significant figures.
            val = np.format_float_positional(val, precision=4, unique=False, fractional=False, trim='k')

            # Write the optimised conditions, objective value and computation time to the results file.
            file.write('----------- {} Underlayers -----------\n'.format(num_underlayers))
            file.write('Thicknesses: {}\n'.format(list(np.round(thicknesses, 1))))
            file.write('SLDs (%): {}\n'.format(list(np.round(slds, 2))))
            file.write('Objective value: {}\n'.format(val))
            file.write('Computation time: {}\n\n'.format(round(end-start, 1)))

if __name__ == '__main__':
    from structures import SymmetricBilayer, SingleAsymmetricBilayer

    sample = SymmetricBilayer()
    optimiser = Optimiser(sample)

    total_time = 1000
    angle_bounds = (0.2, 4)
    #_angle_results(optimiser, total_time, angle_bounds)

    angle_splits = [(0.5, 100, 0.06), (2.3, 100, 0.94)]
    contrast_bounds = (-0.56, 6.36)
    #_contrast_results(optimiser, total_time, angle_splits, contrast_bounds)
    
    angle_times = [(0.5, 100, 5), (2.3, 100, 95)]
    contrasts = [6.36]
    thick_bounds = (0, 500)
    sld_bounds = (1, 9)
    _underlayer_results(optimiser, angle_times, contrasts, thick_bounds, sld_bounds)
