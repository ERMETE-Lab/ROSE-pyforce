[![Reference Paper 1](https://img.shields.io/badge/Reference%20Paper%201-10.1016/j.apm.2024.06.040-gray?labelColor=blue&style=flat&link=https://doi.org/10.1016/j.apm.2024.06.040)](https://doi.org/10.1016/j.apm.2024.06.040) [![Reference Paper 2](https://img.shields.io/badge/Reference%20Paper%202-10.1016/j.nucengdes.2024.113105-gray?labelColor=blue&style=flat&link=https://www.sciencedirect.com/science/article/pii/S002954932400205X)](https://www.sciencedirect.com/science/article/pii/S002954932400205X)

[![Docs](https://img.shields.io/badge/Docs-green?style=flat&link=https://ermete-lab.github.io/ROSE-pyforce/intro.html)](https://ermete-lab.github.io/ROSE-pyforce/intro.html) [![Tutorials](https://img.shields.io/badge/Tutorials-red?style=flat&link=https://ermete-lab.github.io/ROSE-pyforce/tutorials.html)](https://ermete-lab.github.io/ROSE-pyforce/tutorials.html) [![Zenodo](https://img.shields.io/badge/Zenodo-purple?style=flat&link=https://zenodo.org/records/15705990)](https://zenodo.org/records/15705990)


[![Testing pyforce](https://github.com/Steriva/ROSE-pyforce/actions/workflows/testing.yml/badge.svg)](https://github.com/Steriva/ROSE-pyforce/actions/workflows/testing.yml) [![JOSS draft paper](https://github.com/ERMETE-Lab/ROSE-pyforce/actions/workflows/draft-pdf.yml/badge.svg)](https://github.com/ERMETE-Lab/ROSE-pyforce/actions/workflows/draft-pdf.yml)

**pyforce: Python Framework data-driven model Order Reduction for multi-physiCs problEms**

## Description

*pyforce* is a Python package implementing Data-Driven Reduced Order Modelling (DDROM) techniques for applications to multi-physics problems, mainly set in the **Nuclear Engineering** world. The package is part of the **ROSE (Reduced Order modelling with data-driven techniques for multi-phySics problEms)**: mathematical algorithms aimed at reducing the complexity of multi-physics models (for nuclear reactors applications), at searching for optimal sensor positions and at integrating real measures to improve the knowledge on the physical systems.

With respect to the previous original implementation based on [dolfinx](https://github.com/FEniCS/dolfinx) package (v0.6.0), version 1.0.0 of *pyforce* has been completely re-written using `pyvista` as backend for mesh importing, computing integrals, and visualisation of results; in addition, functions are stored as `numpy` arrays, improving the ease of use of the package. **This choice allows to use *pyforce* with any software solver able to export results in VTK format.**
