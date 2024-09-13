# Welcome to *pyforce*'s documentation

**Authors**: Stefano Riva, Carolina Introini, Antonio Cammi

![pyforce](images/immy_pyforce2.png)

[![Reference Paper 1](https://img.shields.io/badge/Reference%20Paper%201-10.1016/j.apm.2024.06.040-gray?labelColor=blue&style=flat&link=https://doi.org/10.1016/j.apm.2024.06.040)](https://doi.org/10.1016/j.apm.2024.06.040) [![Reference Paper 2](https://img.shields.io/badge/Reference%20Paper%202-10.1016/j.nucengdes.2024.113105-gray?labelColor=blue&style=flat&link=https://www.sciencedirect.com/science/article/pii/S002954932400205X)](https://www.sciencedirect.com/science/article/pii/S002954932400205X)


## Description

*pyforce* (Python Framework for data-driven model Order Reduction of multi-physiCs problEms) is a Python package implementing some Data-Driven Reduced Order Modelling (DDROM) techniques for applications to multi-physics problems, mainly set in the **nuclear engineering** world. These techniques have been implemented upon the [dolfinx](https://github.com/FEniCS/dolfinx) package (currently v0.6.0), part of the [FEniCSx](https://fenicsproject.org/) project, to handle mesh generation, integral calculation and functions storage. The package is part of the **ROSE (Reduced Order modelling with data-driven techniques for multi-phySics problEms)**which is one of the main research topics investigated at the [ERMETE-Lab](https://github.com/ERMETE-Lab#reduced-order-modelling-with-data-driven-techniques-for-multi-physics-problems-rose-): in particular, the focus of the research activities is on mathematical algorithms aimed at reducing the complexity of multi-physics models with a focus on nuclear reactor applications, searching for optimal sensor positions and integrating experimental data to improve the knowledge on the physical systems.

At the moment, the following techniques have been implemented:

- **Proper Orthogonal Decomposition** with Projection and Interpolation for the Online Phase
- **Generalised Empirical Interpolation Method**, either with or without Tikhonov's regularisation
- **Parameterised-Background Data-Weak formulation**
- an **Indirect Reconstruction** algorithm to reconstruct non-observable fields

This package is aimed to be a valuable tool for other researchers, engineers, and data scientists working in various fields, not only restricted to the Nuclear Engineering world. This documentation includes a brief introduction to the world of Reduced Order Modelling and dimensionality reduction, the API documentation and some examples of how to use the various modules of the package.

This work has been carried out at the [Nuclear Reactors Group - ERMETE Lab](https://github.com/ERMETE-Lab) at [Politecnico di Milano](https://polimi.it), under the supervision of Prof. Antonio Cammi.

---

## How to cite pyforce

If you are going to use *pyforce* in your research work, please cite the following articles:

1. Stefano Riva, Carolina Introini, and Antonio Cammi, “Multi-physics model bias correction with data-driven reduced order techniques: Application to nuclear case studies,” Applied Mathematical Modelling, vol. 135, pp. 243–268, 2024. [https://doi.org/10.1016/j.apm.2024.06.040](https://doi.org/10.1016/j.apm.2024.06.040).
2. Antonio Cammi, Stefano Riva, Carolina Introini, Lorenzo Loi, and Enrico Padovani. Data-driven model order reduction for sensor positioning and indirect reconstruction with noisy data: Application to a circulating fuel reactor. Nuclear Engineering and Design, 421:113105, 2024. doi:[https://doi.org/10.1016/j.nucengdes.2024.113105](https://doi.org/10.1016/j.nucengdes.2024.113105).

For LaTeX users:

```bibtex

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
