# Installation notes

**pyforce** has been tested on MacOS and Linux machines with **Python3.10**. For Windows, it is suggested to use the Windows Subsystem for Linux (WSL).

## Dependencies
The *pyforce* package requires the following dependencies:

```
numpy
scipy
matplotlib
pyvista
h5py
ipykernel
tqdm
fluidfoam
trame-vtk
scikit-learn
setuptools
```

## Set up a conda environment for *pyforce*

**Currently *pyforce* can only be obtained by directly cloning the repository (not in PyPI or conda repository).**
**This procedure is the suggested one for both users and developers.**

To ensure a quicker and easier installation, it is suggested to change the *conda-solver* to `libmamba`:
```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

At first, clone the repository
```bash
git clone https://github.com/ERMETE-Lab/ROSE-pyforce.git
cd ROSE-pyforce
```

*If you want to install the development version*, clone the repo from Steriva's account

```bash
git clone --branch development --single-branch https://github.com/Steriva/ROSE-pyforce.git
cd ROSE-pyforce
```

At this point, you choose one of the two following procedures. The first one is the quickest and easiest, and it is suggested if you want to install *pyforce* in an existing conda environment, consisting in a direct `pip` installation; the second one is more robust, it uses the `environment.yml` file provided in the repository which creates a conda environment with all the necessary dependencies. The second one is suggested if you encounter issues with the first procedure.


### 1. Using `pip` directly

Given an existing conda environment, you can install *pyforce* using `pip` as
```bash
python -m pip install rose-pyforce/
```

### 2. Using `environment.yml`

Once the repository has been cloned, you can directly install the dependencies by creating a new conda environment using the provided `environment.yml` file:

```bash
conda env create -f rose-pyforce/environment.yml
conda activate pyforce-env
```

Then, install the package using `pip`
```bash
python -m pip install rose-pyforce/
```

This procedure can solve issues related to the installation of some dependencies, especially `pyvista`.
