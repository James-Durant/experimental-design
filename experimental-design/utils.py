import matplotlib.pyplot as plt
import numpy as np
import os, refl1d.model, refl1d.probe, refl1d.experiment

from typing import Optional, List
from numpy.typing import ArrayLike

from refnx.reflect import Structure, ReflectModel
from refnx.analysis import Parameter, Objective, CurveFitter

from dynesty import NestedSampler, DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty import utils as dyfunc

class Sampler:
    """Samples an objective using MCMC or nested sampling.

    Attributes:
        objective (refnx.analysis.Objective): objective to sample.
        ndim (int): number of parameters in objective.
        sampler_MCMC (refnx.analysis.CurveFitter): sampler for MCMC sampling.
        sampler_nested_static (dynesty.NestedSampler): static nested sampler.
        sampler_nested_dynamic (dynesty.DynamicNestedSampler): dynamic nested sampler.

    """
    def __init__(self, objective: Objective) -> None:
        self.objective = objective
        self.ndim = len(self.objective.varying_parameters())
        self.sampler_MCMC = CurveFitter(self.objective)
        self.sampler_nested_static = NestedSampler(self.logl, self.objective.prior_transform, self.ndim)
        self.sampler_nested_dynamic = DynamicNestedSampler(self.logl, self.objective.prior_transform, self.ndim)

    def sample_MCMC(self, burn: int=400, steps: int=30, nthin: int=100, fit_first: bool=True,
                    verbose: bool=True, show_fig: bool=True) -> Optional[plt.Figure]:
        """Samples the objective using MCMC sampling.

        Args:
            burn (int): number of samples to use for burn-in period.
            steps (int): number of steps to use for main sampling stage.
            nthin (int): amount of thinning to use for main sampling stage.
            fit_first (bool): whether to fit before sampling.
            verbose (bool): whether to display progress when sampling.
            show_fig (bool): whether to create and return a corner plot.

        Returns:
            (matplotlib.pyplot.Figure, optional): MCMC sampling corner plot.

        """
        # Initially fit with differential evolution if requested.
        if fit_first:
           self.sampler_MCMC.fit('differential_evolution', verbose=verbose)

        # Burn-in period.
        self.sampler_MCMC.sample(burn, verbose=verbose)

        # Main sampling stage.
        self.sampler_MCMC.reset()
        self.sampler_MCMC.sample(steps, nthin=nthin, verbose=verbose)

        # Return the sampling corner plot if requested.
        if show_fig:
            return self.objective.corner()

    def sample_nested(self, dynamic: bool=False, verbose: bool=True, show_fig: bool=True) -> Optional[plt.Figure]:
        """Samples the objective using static or dynamic nested sampling.

        Args:
            dynamic (bool): whether to use static or dynamic nested sampling.
            verbose (bool): whether to display progress when sampling.
            show_fig (bool): whether to create and return a corner plot.

        Returns:
            (matplotlib.pyplot.Figure, optional): nested sampling corner plot.

        """
        # Sample using static or dynamic nested sampling.
        if dynamic:
            # Weighting is entirely on the posterior (0 weight on evidence).
            self.sampler_nested_dynamic.run_nested(print_progress=verbose, wt_kwargs={'pfrac': 1.0})
            results = self.sampler_nested_dynamic.results
        else:
            self.sampler_nested_static.run_nested(print_progress=verbose)
            results = self.sampler_nested_static.results

        # Calculate the parameter means.
        weights = np.exp(results.logwt - results.logz[-1])
        mean, _ = dyfunc.mean_and_cov(results.samples, weights)

        # Update objective to use mean parameter values.
        self.logl(mean)

        # Return the sampling corner plot if requested.
        if show_fig:
            fig, _ = dyplot.cornerplot(results, color='blue', quantiles=None, show_titles=True,
                                       max_n_ticks=3, truths=np.zeros(self.ndim),truth_color='black')

            # Label axes with parameter names.
            axes = np.reshape(np.array(fig.get_axes()), (self.ndim, self.ndim))
            parameters = self.objective.varying_parameters()
            for i in range(1, self.ndim):
                for j in range(self.ndim):
                    if i == self.ndim-1:
                        axes[i,j].set_xlabel(parameters[j].name)
                    if j == 0:
                        axes[i,j].set_ylabel(parameters[i].name)

            axes[self.ndim-1, self.ndim-1].set_xlabel(parameters[-1].name)
            return fig

    def logl(self, x: ArrayLike) -> float:
        """Calculates the log-likelihood of the parameters `x` against the model.

        Args:
            x (numpy.ndarray): parameter values.

        Returns:
            (float): log-likelihood of given parameter values.

        """
        # Update the model with given parameter values.
        for i, parameter in enumerate(self.objective.varying_parameters()):
            parameter.value = x[i]
        return self.objective.logl()

def vary_structure(structure, bound_size=0.2):
    params = []
    for component in structure[1:-1]:
        if isinstance(structure, Structure):
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
            sld.range(sld.value*(1-bound_size), sld.value*(1+bound_size))
            params.append(sld)
            
            thick = component.thickness
            thick.range(thick.value*(1-bound_size), thick.value*(1+bound_size))
            params.append(thick)

    return params

def fisher(qs: List[ArrayLike], xi: List[Parameter], counts: List[ArrayLike],
           models: List[ReflectModel], step: float=0.005) -> ArrayLike:
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
    if isinstance(model, ReflectModel):
        return model(q)
        
    if isinstance(model, refl1d.experiment.Experiment):
        scale, bkg = model.probe.intensity, model.probe.background
        
        probe = refl1d.probe.QProbe(q, np.zeros_like(q), intensity=scale, background=bkg)
        experiment = refl1d.experiment.Experiment(probe=probe, sample=model.sample)

        return experiment.reflectivity(resolution=True)[1]

def save_plot(fig: plt.Figure, save_path: str, filename: str) -> None:
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
