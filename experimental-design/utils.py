import matplotlib.pyplot as plt
import numpy as np
import os

from typing import Optional, Tuple, List
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

def vary_structure(structure: Structure, random_init: bool=False, bound_size: float=0.2,
                   vary_sld: bool=True, vary_thick: bool=True,
                   vary_rough: bool=False, vary_substrate: bool=False) -> Structure:
    """Vary the parameters of each layer of a given `structure` and optionally,
       initialise these values to random values within their bounds.

    Args:
        structure (refnx.reflect.Structure): structure to vary.
        random_init (bool): whether to randomly initialise parameters.
        bound_size (float): size of bounds to place on parameters.
        vary_sld (bool): whether to vary structure's layers' SLDs.
        vary_thick (bool): whether to vary structure's layers' thicknesses.
        vary_rough (bool): whether to vary structure's layers' roughnesses.
        vary_substrate (bool): whether to vary substrate roughness.

    Returns:
        (refnx.reflect.Structure): reference to input `structure`.
        params

    """
    params = []
    # Skip over air/water and substrate.
    for component in structure.components[1:-1]:
        # Vary each layers' SLD, thickness and roughness if requested.
        if vary_sld:
            sld_bounds = (component.sld.real.value*(1-bound_size),
                          component.sld.real.value*(1+bound_size))
            component.sld.real.setp(vary=True, bounds=sld_bounds)
            params.append(component.sld.real)
            
            # Set parameter to an arbitrary initial value within its bounds.
            if random_init:
                component.sld.real.value = np.random.uniform(*sld_bounds)

        if vary_thick:
            thick_bounds = (component.thick.value*(1-bound_size),
                            component.thick.value*(1+bound_size))
            component.thick.setp(vary=True, bounds=thick_bounds)
            params.append(component.thick)
            
            if random_init:
                component.thick.value = np.random.uniform(*thick_bounds)

        if vary_rough:
            rough_bounds = (component.rough.value*(1-bound_size),
                            component.rough.value*(1+bound_size))
            component.rough.setp(vary=True, bounds=rough_bounds)
            params.append(component.rough)
            
            if random_init:
                component.rough.value = np.random.uniform(*rough_bounds)

    # Vary the substrate's roughness.
    if vary_substrate:
        component = structure.components[-1]
        rough_bounds = (component.rough.value*(1-bound_size),
                        component.rough.value*(1+bound_size))
        component.rough.setp(vary=True, bounds=rough_bounds)
        params.append(component.rough)
        
        if random_init:
            component.rough.value = np.random.uniform(*rough_bounds)

    return structure, params

def fisher_single_contrast(q: ArrayLike, xi: List[Parameter], counts: ArrayLike,
                           model: ReflectModel, step: float=0.005) -> ArrayLike:
    """Calculates the FI matrix for a single `model`.

    Args:
        q (numpy.ndarray): momentum transfer values.
        xi (list): varying parameters.
        counts (numpy.ndarray): incident neutron counts for each Q value.
        model (refnx.reflect.ReflectModel): model for calculating gradients.
        step (float): step size to take when calculating gradient.

    Returns:
        (numpy.ndarray): FI matrix for the given model and data.

    """
    n = len(q) # Number of data points.
    m = len(xi) # Number of parameters.
    J = np.zeros((n,m))

    # Calculate the gradient of the model reflectivity with every model
    # parameter for every model data point.
    for i in range(m):
        parameter = xi[i]
        old = parameter.value

        # Calculate the model reflectance for the first side of the gradient.
        x1 = parameter.value = old*(1-step)
        y1 = model(q)

        # Calculate the model reflectance for the second side of the gradient.
        x2 = parameter.value = old*(1+step)
        y2 = model(q)

        parameter.value = old # Reset the parameter.

        J[:,i] = (y2-y1) / (x2-x1) # Calculate the gradient.

    # Calculate the FI matrix using the equations from the paper.
    M = np.diag(counts / model(q), k=0)
    return np.dot(np.dot(J.T, M), J)

def fisher_multiple_contrasts(qs: List[ArrayLike], xi: List[Parameter], counts: List[ArrayLike],
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
        y1 = np.concatenate([model(q) for q, model in list(zip(qs, models))])

        # Calculate the reflectance for each model for the second side of the gradient.
        x2 = parameter.value = old*(1+step)
        y2 = np.concatenate([model(q) for q, model in list(zip(qs, models))])

        parameter.value = old # Reset the parameter.

        J[:,i] = (y2-y1) / (x2-x1) # Calculate the gradient.

    # Calculate the reflectance for each model for the given Q values.
    r = np.concatenate([model(q) for q, model in list(zip(qs, models))])

    # Calculate the FI matrix using the equations from the paper.
    M = np.diag(np.concatenate(counts) / r, k=0)
    return np.dot(np.dot(J.T, M), J)

def plot_objective(objective: Objective, colour: str='black',
                   label: bool=False) -> Tuple[plt.Figure, plt.Axes]:
    """Plots the fit of a given `objective` against the objective's data.

    Args:
        objective (refnx.analysis.Objective): objective to plot.
        colour (str): colour to use for objective's data points.
        label (bool): whether to use structure's name in plot's legend.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing plotted objective.
        ax (matplotlib.pyplot.Axes): axis containing plotted objective.

    """
    # Plot the reflectivity data.
    fig = plt.figure(figsize=[9,7], dpi=600)
    ax = fig.add_subplot(111)

    # Plot the reflectivity data (Q, R, dR).
    q, r, dr = q = objective.data.x, objective.data.y, objective.data.y_err
    ax.errorbar(q, r, dr, color=colour, marker='o', ms=3, lw=0,
                elinewidth=1, capsize=1.5, label=label)

    # Plot the fit.
    ax.plot(q, objective.model(q), color='red', zorder=20)

    ax.set_xlabel('$\mathregular{Q\ (Å^{-1})}$', fontsize=11, weight='bold')
    ax.set_ylabel('Reflectivity (arb.)', fontsize=11, weight='bold')
    ax.set_yscale('log')
    ax.set_ylim(1e-7, 2)
    return fig, ax

def plot_objectives(objectives: List[Objective], label: bool=True) -> Tuple[plt.Figure, plt.Axes]:
    """Plots fits of `objectives` against the objectives' data.

    Args:
        objectives (list): objectives to plot.
        label (bool): whether to include a legend in plot.

    Returns:
        fig (matplotlib.pyplot.Figure): figure containing plotted objectives.
        ax (matplotlib.pyplot.Axes): axis containing plotted objectives.

    """
    # Get the figure and axis by plotting the first objective by itself.
    fig, ax = plot_objective(objectives[0], None, label)

    # Plot the remaining `objectives` on the same axis.
    for objective in objectives[1:]:
        q, r = objective.data.x, objective.data.y
        r_error = objective.data.y_err

        # Plot the reflectivity data.
        ax.errorbar(q, r, r_error, marker='o', ms=3, lw=0, elinewidth=1, capsize=1.5,
                    label=objective.model.structure.name if label else None)
        # Plot the fit.
        ax.plot(q, objective.model(q), color='red', zorder=20)

    if label: # If labelling, create the legend.
        ax.legend()

    return fig, ax

def save_plot(fig: plt.Figure, save_path: str, file_name: str) -> None:
    """Saves a figure to a given directory.

    Args:
        fig (matplotlib.pyplot.Figure): figure to save.
        save_path (str): path to directory to save the figure to.
        file_name (str): name of file to save the plot as.

    """
    # Create the directory if not present.
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, file_name+'.png')
    fig.savefig(file_path, dpi=600)
