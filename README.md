[![DOI](https://zenodo.org/badge/366323997.svg)](https://zenodo.org/badge/latestdoi/366323997)

# experimental-design
## About the Project
**For the original repository that this work is based on, see [fisher-information](https://github.com/James-Durant/fisher-information)**.

Using the Fisher information (FI), the design of neutron reflectometry experiments can be optimised, leading to greater confidence in parameters of interest and better use of experimental time. This repository contains the [code](/experimental-design), [data](/experimental-design/data) and [results](/experimental-design/results) for optimising the design of a wide range of reflectometry experiments.

Please refer to the [notebooks](/notebooks) directory for an introduction.

### Citation
Please cite the following article if you intend on including elements of this work in your own publications:
> Durant, J. H., Wilkins, L. and Cooper, J. F. K. Optimizing experimental design in neutron reflectometry. _J. Appl. Cryst_. **55** (2021).

Or with BibTeX as:
```
@article{Durant2022,
   author    = {Durant, J. H. and Wilkins, L. and Cooper, J. F. K.},
   doi       = {10.1107/S1600576722003831},
   journal   = {Journal of Applied Crystallography},
   month     = {Aug},
   number    = {4},
   pages     = {},
   publisher = {International Union of Crystallography ({IUCr})},
   title     = {{Optimizing experimental design in neutron reflectometry}},
   url       = {https://doi.org/10.1107/S1600576722003831},
   volume    = {55},
   year      = {2022}
}
```

For the results presented in this article, see [notebooks](/notebooks), and for the figures, see [figures](/figures).

## Installation
1. To replicate the development environment with the [Anaconda](https://www.anaconda.com/products/individual) distribution, first create an empty conda environment by running: <br /> ```conda create --name experimental-design python=3.8.3```

2. To activate the environment, run: ```conda activate experimental-design```

3. Install pip by running: ```conda install pip```

4. Run the following to install the required packages from the [requirements.txt](/requirements.txt) file: <br />
   ```pip install -r requirements.txt```

You should now be able to run the code. Please ensure you are running a version of Python >= 3.8.0 \
If you are running an old version of Anaconda, you may need to reinstall with a newer version for this.

## Contact
Jos Cooper - jos.cooper@stfc.ac.uk \
James Durant - james.durant@warwick.ac.uk \
Lucas Wilkins - lucas@lucaswilkins.com

## Acknowledgements
We thank Luke Clifton for his assistance and expertise in fitting the lipid monolayer and lipid bilayer data sets.

## License
Distributed under the GPL-3.0 License. See [license](/LICENSE) for more information.
