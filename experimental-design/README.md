## Directories
* [data](/experimental-design/data) - Directbeam files and experimentally-measured data for a selection of samples of varying complexity.
* [models](/experimental-design/models) - Model definitions and fits for the aforementioned samples.
* [results](/experimental-design/results) - Results for the aforementioned samples.

## Code
* [angles.py](/experimental-design/angles.py) - Optimises and visualises the choice of measurement angle(s) for a collection of samples of varying complexity.
* [contrasts.py](/experimental-design/contrasts.py) - Optimises and visualises the choice of contrast for the [DMPC](/experimental-design/results/DMPC_bilayer) and [DPPC/RaLPS](/experimental-design/results/DPPC_RaLPS_bilayer) bilayer models.
* [kinetics.py](/experimental-design/kinetics.py) - Optimises and visualises the choice of measurement angle and contrast for the [DPPG](/experimental-design/results/DPPG_monolayer) monolayer model degrading over time.
* [magnetism.py](/experimental-design/magnetism.py) - Optimises and visualises the sample design of the magnetic [YIG](/experimental-design/results/YIG_sample) sample.
* [optimise.py](/experimental-design/optimise.py) - Contains code for optimising the choice of measurement angle(s), counting time(s), contrast(s) and underlayer(s).
* [simulate.py](/experimental-design/simulate.py) - Contains code for simulating experiments using a [directbeam](/experimental-design/data/directbeams) file of incident neutron flux as a function of wavelength.
* [underlayers.py](/experimental-design/underlayers.py) - Optimises and visualises the choice of underlayer thickness(es) and SLD(s) for the [DMPC](/experimental-design/results/DMPC_bilayer) and [DPPC/RaLPS](/experimental-design/results/DPPC_RaLPS_bilayer) bilayer models.
* [utils.py](/experimental-design/utils.py) - Contains miscellaneous code for calculating the Fisher information, nested sampling, and saving plots.
* [visualise.py](/experimental-design/visualise.py) - Contains code for visualising the choice of measurement angle(s), counting time(s), contrast(s) and underlayer(s).
