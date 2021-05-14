import numpy as np
import os

from dynesty import NestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

import refl1d.probe, refl1d.model, refl1d.experiment
import refnx.reflect, refnx.analysis
import bumps.fitproblem

class Sampler:
    def __init__(self, objective):  
        self.objective = objective
        
        if isinstance(objective, refnx.analysis.BaseObjective):
            self.params = objective.varying_parameters()
            logl = objective.logl
            prior_transform = objective.prior_transform
            
        elif isinstance(objective, bumps.fitproblem.BaseFitProblem): 
            self.params = self.objective._parameters
            logl = self.logl_refl1d
            prior_transform = self.prior_transform_refl1d
            
        else:
            raise RuntimeError('invalid objective/fitproblem given')

        self.ndim = len(self.params)
        self.sampler_nested = NestedSampler(logl, prior_transform, self.ndim)

    def logl_refl1d(self, x):
        self.objective.setp(x)
        return -self.objective.model_nllf()

    def prior_transform_refl1d(self, u):
        x = [param.bounds.put01(u[i]) for i, param in enumerate(self.params)]
        return np.asarray(x)

    def sample(self, verbose=True):
        self.sampler_nested.run_nested(print_progress=verbose)
        results = self.sampler_nested.results

        # Calculate the parameter means.
        weights = np.exp(results.logwt - results.logz[-1])
        mean, _ = dyfunc.mean_and_cov(results.samples, weights)
        
        for i, param in enumerate(self.params):
            param.value = mean[i]

        self.corner(results)

    def corner(self, results):
        fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None,
                                   show_titles=True, max_n_ticks=3,
                                   truths=np.zeros(self.ndim),
                                   truth_color='black')

        # Label axes with parameter names.
        axes = np.reshape(np.array(fig.get_axes()), (self.ndim, self.ndim))
        for i in range(1, self.ndim):
            for j in range(self.ndim):
                if i == self.ndim-1:
                    axes[i,j].set_xlabel(self.params[j].name)
                if j == 0:
                    axes[i,j].set_ylabel(self.params[i].name)

        axes[self.ndim-1, self.ndim-1].set_xlabel(self.params[-1].name)
        return fig

def vary_structure(structure, bound_size=0.2):
    params = []
    for component in structure[1:-1]:
        if isinstance(structure, refnx.reflect.Structure):
            sld = component.sld.real
            sld_bounds = (sld.value*(1-bound_size), sld.value*(1+bound_size))
            sld.setp(vary=True, bounds=sld_bounds)
            params.append(sld)
            
            thick = component.thick
            thick_bounds = (thick.value*(1-bound_size), thick.value*(1+bound_size))
            thick.setp(vary=True, bounds=thick_bounds)
            params.append(thick)
            
        elif isinstance(structure, refl1d.model.Stack): 
            sld = component.material.rho
            sld.pmp(bound_size*100)
            params.append(sld)
            
            thick = component.thickness
            thick.pmp(bound_size*100)
            params.append(thick)

    return params

def fisher(qs, xi, counts, models, step=0.005):
    """Calculates the FI matrix for multiple `models` containing parameters, `xi`.

    Args:
        qs (list): momentum transfer values for each model.
        xi (list): varying model parameters.
        counts (list): incident neutron counts corresponding to each Q value.
        models (list): models to calculate gradients with.
        step (float): step size to take when calculating gradient.

    Returns:
        (numpy.ndarray): FI matrix for the given models and data.

    """
    n = sum(len(q) for q in qs) # Number of data points.
    m = len(xi) # Number of parameters.
    J = np.zeros((n,m))

    # Calculate the gradient of the model reflectivity with every model
    # parameter for every model data point.
    for i in range(m):
        parameter = xi[i]
        old = parameter.value

        # Calculate the reflectance for each model for the first side of the gradient.
        x1 = parameter.value = old*(1-step)
        y1 = np.concatenate([reflectivity(q, model) for q, model in list(zip(qs, models))])

        # Calculate the reflectance for each model for the second side of the gradient.
        x2 = parameter.value = old*(1+step)
        y2 = np.concatenate([reflectivity(q, model) for q, model in list(zip(qs, models))])

        parameter.value = old # Reset the parameter.

        J[:,i] = (y2-y1) / (x2-x1) # Calculate the gradient.

    # Calculate the reflectance for each model for the given Q values.
    r = np.concatenate([reflectivity(q, model)for q, model in list(zip(qs, models))])

    # Calculate the FI matrix using the equations from the paper.
    M = np.diag(np.concatenate(counts) / r, k=0)
    return np.dot(np.dot(J.T, M), J)

def reflectivity(q, model):
    if isinstance(model, refnx.reflect.ReflectModel):
        return model(q)
        
    if isinstance(model, refl1d.experiment.Experiment):
        scale, bkg, dq = model.probe.intensity, model.probe.background, model.probe.dQ
        probe = refl1d.probe.QProbe(q, dq, intensity=scale, background=bkg)
        
        experiment = refl1d.experiment.Experiment(probe=probe, sample=model.sample)
        return experiment.reflectivity(resolution=True)[1]

def save_plot(fig, save_path, filename):
    """Saves a figure to a given directory.

    Args:
        fig (matplotlib.pyplot.Figure): figure to save.
        save_path (str): path to directory to save the figure to.
        filename (str): name of file to save the plot as.

    """
    # Create the directory if not present.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, filename+'.png')
    fig.savefig(file_path, dpi=600)

if __name__ == '__main__':
    from structures import simple_sample, refnx_to_refl1d
    from simulate import simulate
    
    sample = refnx_to_refl1d(simple_sample())
    vary_structure(sample)
    
    angle_times = {0.7: (70, 5),
                   2.0: (70, 20)}
    
    model, data = simulate(sample, angle_times)
    
    objective = bumps.fitproblem.FitProblem(model)
    
    sampler = Sampler(objective)
    sampler.sample()
    