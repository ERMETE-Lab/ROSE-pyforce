<p align="center">
  <a href="https://github.com/ERMETE-Lab" target="_blank" >
    <img alt="pyforce" src="./images/immy_pyforce2.png" width="600" />
  </a>
</p>

<p align="center">
  <a href="https://doi.org/10.21105/joss.06950">
    <img src="https://img.shields.io/badge/JOSS%20(Original%20Version)-10.21105/joss.06950-brightgreen?style=flat&logo=journal-of-open-source-software&logoColor=white" alt="JOSS Paper">
  </a>
  <a href="https://doi.org/10.1016/j.nucengdes.2024.113105">
    <img src="https://img.shields.io/badge/Ref%20Paper%201-Nucl.%20Eng.%20Des.-gray?labelColor=blue&style=flat" alt="Reference Paper 1">
  </a>
  <a href="https://doi.org/10.1016/j.apm.2024.06.040">
    <img src="https://img.shields.io/badge/Ref%20Paper%202-Appl.%20Math.%20Mod.-gray?labelColor=blue&style=flat" alt="Reference Paper 2">
  </a>
</p>

<p align="center">
  <a href="https://ermete-lab.github.io/ROSE-pyforce/intro.html">
    <img src="https://img.shields.io/badge/Docs-Read%20the%20Docs-green?style=flat&logo=readthedocs&logoColor=white" alt="Docs">
  </a>
  <a href="https://zenodo.org/records/15705990">
    <img src="https://img.shields.io/badge/Datasets-Zenodo-purple?style=flat&logo=zenodo&logoColor=white" alt="Zenodo">
  </a>
  <a href="https://github.com/Steriva/ROSE-pyforce/actions/workflows/testing.yml">
    <img src="https://github.com/Steriva/ROSE-pyforce/actions/workflows/testing.yml/badge.svg" alt="Testing">
  </a>
</p>

**pyforce: PYthon Framework for data-driven model Order Reduction of multi-physiCs problems**

- [Description](#description)
- [How to cite *pyforce*](#how-to-cite-pyforce)
  - [Selected works using *pyforce*](#selected-works-using-pyforce)
- [Installation](#installation)
- [Tutorials](#tutorials)
- [Authors and contributions](#authors-and-contributions)
- [Community Guidelines](#community-guidelines)
  - [Contribute to the Software](#contribute-to-the-software)
  - [Reporting Issues or Problems](#reporting-issues-or-problems)
  - [Seeking Support](#seeking-support)

## Description

*pyforce* is a Python package implementing Data-Driven Reduced Order Modelling (DDROM) techniques for applications to multi-physics problems, mainly set in the **Nuclear Engineering** world. The package is part of the **ROSE (Reduced Order modelling with data-driven techniques for multi-phySics problEms)**: mathematical algorithms aimed at reducing the complexity of multi-physics models (for nuclear reactors applications), at searching for optimal sensor positions and at integrating real measures to improve the knowledge on the physical systems.

With respect to the previous original implementation based on [dolfinx](https://github.com/FEniCS/dolfinx) package (v0.6.0), version 1.0.0 of *pyforce* has been completely re-written using `pyvista` as backend for mesh importing, computing integrals, and visualisation of results; in addition, functions are stored as `numpy` arrays, improving the ease of use of the package. **This choice allows to use *pyforce* with any software solver able to export results in VTK format.**

The techniques implemented here follow the same underlying idea expressed in the following figure: in the offline (training) phase, a dimensionality reduction process retrieves a reduced coordinate system onto which encodes the information of the mathematical model; the sensor positioning algorithm then uses this set to select the optimal location of sensors according to some optimality criterion, which depends on the adopted algorithm. In the online phase, the DA process begins, retrieving a novel set of reduced variables and then computing the reconstructed state through a decoding step [Riva et al. (2024)](https://doi.org/10.1016/j.apm.2024.06.040).

<p align="center">
  <img alt="DDROMstructure" src="images/tie_frighter.svg" width="850" />
  </a>
</p>

At the moment, the following techniques have been implemented:

- **Singular Value Decomposition** (randomised), with Projection and Interpolation for the Online Phase
- **Proper Orthogonal Decomposition** with Projection and Interpolation for the Online Phase
- **Empirical Interpolation Method**, either regularised with Tikhonov or not
- **Generalised Empirical Interpolation Method**, either regularised with Tikhonov or not
- **Parameterised-Background Data-Weak formulation**
- **SGreedy** algorithm for optimal sensor positioning
- an **Indirect Reconstruction** algorithm to reconstruct non-observable fields

This package is aimed to be a valuable tool for other researchers, engineers, and data scientists working in various fields, not only restricted in the Nuclear Engineering world.

**⚠️ Important Note on Versions**
The reference paper published in JOSS [1] describes the original implementation of *pyforce* (v0.1.3), which was built upon the `dolfinx` FEM framework. **This repository hosts the new major version (v1.0.0+)**, which has been completely re-architected using `pyvista` and `numpy`. This new standalone architecture removes strict FEM dependencies, allowing *pyforce* to process results from **any solver** (e.g., OpenFOAM, Ansys, MOOSE) capable of exporting VTK/H5 files.

## How to cite *pyforce*

If you use *pyforce* in your research, please cite the **JOSS paper** as the primary software reference.
**Note:** While the JOSS paper describes the original `dolfinx`-based implementation, it remains the standard citation for the *pyforce* project until the publication of the new methodology.

1. **[Software Reference]** S. Riva, C. Introini, and A. Cammi, "pyforce: Python Framework for data-driven model Order Reduction of multi-physiCs problems," *Journal of Open Source Software*, vol. 11, no. 117, p. 6950, 2026. [https://doi.org/10.21105/joss.06950](https://doi.org/10.21105/joss.06950)

For the original papers, with applications on nuclear reactors (multiphysics modelling), please also cite:

2. **[Model Bias Correction]** S. Riva, C. Introini, and A. Cammi, "Multi-physics model bias correction...", *Applied Mathematical Modelling*, 2024. [https://doi.org/10.1016/j.apm.2024.06.040](https://doi.org/10.1016/j.apm.2024.06.040)
3. **[Sensor Positioning and Indirect Reconstruction]** A. Cammi, S. Riva, et al., "Data-driven model order reduction...", *Nuclear Engineering and Design*, 2024. [https://doi.org/10.1016/j.nucengdes.2024.113105](https://doi.org/10.1016/j.nucengdes.2024.113105)

For LaTeX users:

```bibtex

@article{pyforce_JOSS,
  doi = {10.21105/joss.06950},
  url = {[https://doi.org/10.21105/joss.06950](https://doi.org/10.21105/joss.06950)},
  year = {2026},
  publisher = {The Open Journal},
  volume = {11},
  number = {117},
  pages = {6950},
  author = {Stefano Riva and Carolina Introini and Antonio Cammi},
  title = {pyforce: Python Framework for data-driven model Order Reduction of multi-physiCs problems},
  journal = {Journal of Open Source Software}
}

@article{RIVA2024_AMM,
title = {Multi-physics model bias correction with data-driven reduced order techniques: Application to nuclear case studies},
journal = {Applied Mathematical Modelling},
volume = {135},
pages = {243-268},
year = {2024},
issn = {0307-904X},
doi = {https://doi.org/10.1016/j.apm.2024.06.040},
url = {https://www.sciencedirect.com/science/article/pii/S0307904X24003196},
author = {Stefano Riva and Carolina Introini and Antonio Cammi},
keywords = {Reduced order modelling, Data driven, Nuclear reactors, Multi-physics, Model correction},
}

@article{CAMMI2024_NED,
title = {Data-driven model order reduction for sensor positioning and indirect reconstruction with noisy data: Application to a Circulating Fuel Reactor},
journal = {Nuclear Engineering and Design},
volume = {421},
pages = {113105},
year = {2024},
issn = {0029-5493},
doi = {https://doi.org/10.1016/j.nucengdes.2024.113105},
url = {https://www.sciencedirect.com/science/article/pii/S002954932400205X},
author = {Antonio Cammi and Stefano Riva and Carolina Introini and Lorenzo Loi and Enrico Padovani},
keywords = {Hybrid Data-Assimilation, Generalized Empirical Interpolation Method, Indirect Reconstruction, Sensors positioning, Molten Salt Fast Reactor, Noisy data},
}

```

### Selected works using *pyforce*

In addition to the above references, here are some selected works where *pyforce* has been used:

- M. Lo Verso, S. Riva, C. Introini, E. Cervi, F. Giacobbo, L. Savoldi, M. Di Prinzio, M. Caramello, L. Barucca, and A. Cammi, “Application of a non-intrusive reduced order modeling approach to magnetohydrodynamics,” Physics of Fluids, vol. 36, p. 107167, 10 2024, [doi.org/10.1063/5.0230708](https://doi.org/10.1063/5.0230708).
- S. Riva, C. Introini, A. Cammi, and J. N. Kutz, “Robust state estimation from partial out-core measurements with shallow recurrent decoder for nuclear reactors,” Progress in Nuclear Energy, vol. 189, p. 105928, 2025, [doi.org/10.1016/j.pnucene.2025.105928](https://doi.org/10.1016/j.pnucene.2025.105928)
- S. Riva, C. Introini, E. Zio, and A. Cammi, “Data-driven reduced order modelling with malfunctioning sensors recovery applied to the molten salt reactor case,” EPJ Nuclear Sci. Technol., vol. 11, p. 55, 2025, [doi.org/10.1051/epjn/2025054](https://doi.org/10.1051/epjn/2025054)
- S. Riva, S. Deanesi, C. Introini, S. Lorenzi, and A. Cammi, “Real-time state estimation of neutron flux in molten salt fast reactors from out-core sparse measurements,” Nuclear Science and Engineering, vol. 0, no. 0, pp. 1–14, 2025, [doi.org/10.1080/00295639.2025.2531477](https://doi.org/10.1080/00295639.2025.2531477).

## Installation

It is recommended to install the package in a conda environment, although it is not strictly required.

The simplest way to install the package is through using `pip`, including all the dependencies.

**Here's how**: at first, clone the repository (this will clone the official one)

```bash
git clone https://github.com/ERMETE-Lab/ROSE-pyforce.git
cd ROSE-pyforce
```

*If you want to install the development version*, clone the repo from Steriva's account

```bash
git clone --branch development --single-branch https://github.com/Steriva/ROSE-pyforce.git
cd ROSE-pyforce
```

then install the package using `pip` (this will work if you already have `python` and `pip` installed, it might not be true for `miniconda` installations):

```bash
python -m pip install rose-pyforce/
```
or equivalently

```bash
cd rose-pyforce
python -m pip install .
```

Another option is also provided adopting the `environment.yml` file as follows (If you face issues with rendering figures with `pyvista`, this might solve the problem):

```bash
conda env create -f pyforce/environment.yml
conda activate pyforce-env
python -m pip install rose-pyforce/
```

The requirements are listed [here](https://github.com/ERMETE-Lab/ROSE-pyforce/blob/main/pyforce/requirements.txt).

## Tutorials
The *pyforce* package is tested on some tutorials available in the [docs](https://ermete-lab.github.io/ROSE-pyforce/tutorials.html), including fluid dynamics, neutronics and multi-physics problems or available in the `docs/Tutorials` folder. Each tutorial includes a Jupyter notebook:

1. First steps with *pyforce*: introduction to the package and its basic features.
2. Introduction to Singular Value Decomposition (SVD) and Proper Orthogonal Decomposition (POD) and application to a fluid dynamics problem using the POD with Interpolation (POD-I) technique.
3. Presentation of (Generalised) Empirical Interpolation Method ((G)EIM) and application to a bouyancy-driven fluid dynamics problem.
4. Sensor positioning with (G)EIM and SGreedy algorithm and application of the Parameterised-Background Data-Weak (PBDW) formulation to a neutronics problem.

In addition to these basic tutorials, some advanced tutorials are also available at `docs/Tutorials/Advanced` folder:

5. Reconstruction of Unobservable fields from temperature measurements using Parameter Estimation + POD with Interpolation and Gaussian Process Regression, applied to a bouyancy-driven fluid dynamics problem.
6. State Estimation in Molten Salt Fast Reactors (MSFR) with failing sensors using GEIM and PBDW techniques.

The snapshots can be downloaded at the following link ... or contact Stefano Riva for further information.

## Authors and contributions

**pyforce** is currently developed and mantained at [Nuclear Reactors Group - ERMETE Lab](https://github.com/ERMETE-Lab) by

- Stefano Riva

under the supervision of Dr. Carolina Introini and Prof. Antonio Cammi.

If interested, please contact stefano.riva@polimi.it, carolina.introini@polimi.it, antonio.cammi@polimi.it

## Community Guidelines

We welcome contributions and feedback from the community! Below are the guidelines on how to get involved:

### Contribute to the Software
If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Implement your changes. If you're adding new features, we kindly ask that you include an example demonstrating how to use them.
3. Submit a pull request for review.

### Reporting Issues or Problems
If you encounter any issues or bugs with *pyforce*, please report them through the GitHub [Issues](https://github.com/ERMETE-Lab/ROSE-pyforce/issues) page. Be sure to include detailed information to help us resolve the problem efficiently.

### Seeking Support
For support, you can either:
- Open a discussion on the GitHub [Discussions](https://github.com/ERMETE-Lab/ROSE-pyforce/discussions) page.
- Send an email directly to: [stefano.riva@polimi.it](mailto:stefano.riva@polimi.it) or [carolina.introini@polimi.it](mailto:carolina.introini@polimi.it)

Thank you for helping improve **pyforce**!
