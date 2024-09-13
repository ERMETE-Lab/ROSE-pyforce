# Installation notes

**pyforce** has been tested on MacOS and Linux machines with **Python3.10**.

## Dependencies
The *pyforce* package requires the following dependencies:

```python
import numpy
import scipy
import matplotlib
import h5py

import pyvista
import gmsh
import dolfinx
import sklearn
import fluidfoam
```

Be sure to install *gmsh* and *gmsh-api* before *dolfinx* (the package has been tested with real mode of the PETSc library). The instructions to install *dolfinx* are available at [https://github.com/FEniCS/dolfinx#binary](https://github.com/FEniCS/dolfinx#binary).

## Set up a conda environment for *pyforce*

**Currently *pyforce* can only be obtained by directly cloning the repository (not in PyPI or conda repository).**

It is suggested to create a conda environment: at first, clone the repository
```bash
git clone https://github.com/ERMETE-Lab/ROSE-pyforce.git
```
create a conda environment using `environment.yml`
```bash
cd ROSE-pyforce
conda env create -f pyforce/environment.yml
```
activate the environment and then install the package using `pip`
```bash
conda activate pyforce-env
python -m pip install pyforce/
```

If the previous procedure encounters any issues, you can adopt a step-by-step approach: start by creating a new conda environment
```bash
conda create --name <env_name>
```
If not already done, add conda-forge to the channels
```bash
conda config --add channels conda-forge
```
After having activate it, install
```bash
conda install python=3.10
```
This provides also *pip* which is necessary to install *gmsh* as
```bash
python -m pip install gmsh gmsh-api
```
Then, *dolfinx* can be installed (real mode for *petsc* is supposed), currently only supports *v0.6.0*,
```bash
conda install fenics-dolfinx=0.6.0 mpich pyvista
```
Just for completeness, if you are to deal with complex numbers use the following command
```bash
conda install fenics-dolfinx=0.6.0 petsc=*=real* mpich pyvista
```
Add the following packages
```bash
conda install meshio scipy tqdm
```
Downgrade the following
```bash
python -m pip install setuptools==62.0.0
conda install numpy=1.23.5
```
Once this is completed, it may be necessary to re-install *gmsh*
```bash
python -m pip install gmsh gmsh-api
```
In the end, the *fluidfoam* ([https://github.com/fluiddyn/fluidfoam](https://github.com/fluiddyn/fluidfoam)) and *scikit-learn* are necessary to import data from OpenFOAM and to integrate Machine Learning (ML) with ROM
```bash
python -m pip install fluidfoam scikit-learn
```
Once all the dependencies have been installed, *pyforce* can be installed using *pip*: clone the repository
```bash
git clone https://github.com/ROSE-Polimi/pyforce.git
```
Change directory to *pyforce* and install using pip
```bash
python -m pip install pyforce/
```
