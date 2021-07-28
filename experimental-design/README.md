# Guide
## Directories
* [data](/experimental-design/data) - Contains directbeam files and experimentally-measured data for a selection of samples of varying complexity.
* [models](/experimental-design/models) - Contains model definitions and fits for the selection of samples.
* [results](/experimental-design/results) - Contains results for the selection of samples.

## Code
* [angles.py](/experimental-design/angles.py) - Optimises and visualises the choice of measurement angle(s) for a collection of samples of varying complexity.
* [contrasts.py](/experimental-design/contrasts.py) - Optimises and visualises the choice of contrast for the DMPC and DPPC/Ra LPS bilayer models.
* [kinetics.py](/experimental-design/kinetics.py) - Optimises and visualises the choice of measurement angle and contrast for the DPPG monolayer model degrading over time.
* [magnetism.py](/experimental-design/magnetism.py) - Optimises and visualises the sample design of the magnetic YIG sample.
* [optimise.py](/experimental-design/optimise.py) - Contains code for optimising the choice of measurement angle(s), counting time(s), contrast(s) and underlayer(s).
* [simulate.py](/experimental-design/simulate.py) - Contains code for simulating experiments using a directbeam file of incident neutron flux as a function of wavelength.
* [underlayers.py](/experimental-design/underlayers.py) - Optimises and visualises the choice of underlayer thickness(es) and SLD(s) for the DMPC and DPPC/Ra LPS bilayer models.
* [utils.py](/experimental-design/utils.py) - Contains miscellaneous code for calculating the Fisher information, nested sampling, and saving plots.
* [visualise.py](/experimental-design/visualise.py) - Contains code for visualising the choice of measurement angle(s), counting time(s), contrast(s) and underlayer(s).
