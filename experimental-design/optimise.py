import numpy as np
import tqdm, sys
sys.path.append('./')

from scipy.optimize import differential_evolution, NonlinearConstraint

from structures import VariableAngle, VariableContrast
from utils import fisher

class Optimiser:
    def __init__(self, sample):
        self.__sample = sample

    def optimise_angle_times(self, num_angles, contrasts=[], total_time=1000, points=100,
                             angle_bounds=(0.2, 2.3), workers=-1):
        assert isinstance(self.__sample, VariableAngle)
        
        bounds = [angle_bounds]*num_angles + [(0, total_time)]*num_angles
        constraints = [NonlinearConstraint(lambda x: sum(x[num_angles:]), total_time, total_time)]
        args = [self.__sample, num_angles, contrasts, points]
    
        res = Optimiser.__optimise(Optimiser._angle_times_func, bounds, constraints, args, workers)
        return res[:num_angles], res[num_angles:]
    
    def optimise_contrasts(self, num_contrasts, angle_times, contrast_bounds=(-0.56, 6.36), workers=-1):
        assert isinstance(self.__sample, VariableContrast)
        
        bounds = [contrast_bounds]*num_contrasts
        args = [self.__sample, num_contrasts, angle_times]
        
        return Optimiser.__optimise(Optimiser._contrasts_func, bounds, [], args, workers)
    
    @staticmethod
    def _angle_times_func(x, sample, num_angles, contrasts, points):
        angle_times = [(x[i], points, x[num_angles+i]) for i in range(num_angles)]
        qs, counts, models = sample.angle_info(angle_times, contrasts)
        g = fisher(qs, sample.parameters, counts, models)
        return -np.linalg.eigvalsh(g)[0]
   
    @staticmethod
    def _contrasts_func(x, sample, num_contrasts, angle_times):
        qs, counts, models = sample.contrast_info(angle_times, x)
        g = fisher(qs, sample.parameters, counts, models)
        return -np.linalg.eigvalsh(g)[0]
    
    @staticmethod
    def __optimise(func, bounds, constraints, args, workers=-1):
        with tqdm.tqdm() as pbar:
            res = differential_evolution(func, bounds, constraints=constraints,
                                         args=args, polish=False, tol=0.001,
                                         updating='deferred', workers=workers,
                                         callback=Optimiser.__callback_wrapper(None, pbar))

        #print('\nNumber of objective calls: {}'.format(res.nfev))
        return res.x

    @staticmethod
    def __callback_wrapper(callback_func, pbar):
        def callback(*args, **kwds):
            pbar.update(1)
            if callback_func is None:
                return None
            else:
                return callback_func(*args, **kwds)
    
        return callback

if __name__ == '__main__':
    from structures import SymmetricBilayer, SingleAsymmetricBilayer
    
    sample = SymmetricBilayer()
    optimiser = Optimiser(sample)
    
    num_angles = 2
    contrasts = [6.36]
    angles, times = optimiser.optimise_angle_times(num_angles, contrasts)
    print('\nAngles: {}'.format(angles))
    print('Times: {}'.format(times))
    
    num_contrasts = 3
    angle_times = [(0.7, 100, 5), (2.3, 100, 95)]
    contrasts = optimiser.optimise_contrasts(num_contrasts, angle_times)
    print('Contrasts: {}'.format(contrasts))
