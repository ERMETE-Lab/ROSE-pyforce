---
title: 'pyforce: Python Framework for data-driven model Order Reduction of multi-physiCs problEms'
tags:
  - Python
  - reduced order modelling
  - nuclear reactors
  - data-driven
  - multi physics
authors:
  - name: Stefano Riva
    orcid: 0000-0001-9997-4101
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted "1, 2")
  - name: Carolina Introini
    orcid: 0000-0003-4682-1683
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Antonio Cammi
    orcid: 0000-0003-1508-5935
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
affiliations:
 - name: Energy Department - Nuclear Engineering Division, Nuclear Reactors Group - ERMETE Lab, Politecnico di Milano
   index: 1
date: 21 May 2024
bibliography: paper.bib

# # Optional fields if submitting to a AAS journal too, see this blog post:
# # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
*pyforce* is a Python library (Python Framework for data-driven model Order Reduction of multi-physiCs problEms) implementing Data-Driven Reduced Order Modelling (DDROM) techniques [@RMP_2024] for applications to multi-physics problems, mainly for the Nuclear Engineering world. These techniques have been implemented upon the dolfinx package [@BarattaEtal2023] (currently v0.6.0), part of the FEniCSx project, to handle mesh generation, integral calculation and functions storage. The package is part of the ROSE (Reduced Order modelling with data-driven techniques for multi-phySics problEms) framework also developed by the authors, which investigates mathematical algorithms aimed at reducing the complexity of multi-physics models with a focus on nuclear reactor applications, at searching for optimal sensor positions and at integrating experimental data to improve the knowledge on the physical systems.

![General scheme of DDROM methods [@RMP_2024].\label{fig:darom}](DA_ROM.pdf){ width=80% }

The techniques implemented here follow the same underlying idea expressed in Figure \autoref{fig:darom}: in the offline (training) phase, a dimensionality reduction process retrieves a reduced coordinate system onto which the information of the mathematical model is encoded; the sensor positioning algorithm then uses this reduced set to select the optimal location of sensors according to some optimality criterion, which depends on the adopted algorithm. In the online phase, the data assimilation process begins, retrieving a novel set of reduced variables and then computing the reconstructed state through a decoding step.

At the moment, the following techniques have been implemented [@DDMOR_CFR;@RMP_2024]:

1. Proper Orthogonal Decomposition (POD) [@quarteroni2015reduced] with Projection and Interpolation [@demo_complete_2019] for the Online Phase
2. Generalised Empirical Interpolation Method (GEIM) [@maday_generalized_2015], either regularised with Tikhonov [@introini_stabilization_2023] or not
3. Parameterised-Background Data-Weak (PBDW) [@maday_parameterized-background_2014]
4. an Indirect Reconstruction [@introini_non-intrusive_2023] algorithm to reconstruct non-observable fields

This package aims to become a valuable tool for other researchers, engineers, and data scientists working in various fields where multi-physics problems play an important role, and its scope of application is not only restricted to the Nuclear Engineering world. The package also includes tutorials showing how to use the library and its main features, ranging from snapshot generation in dolfinx, import and mapping from OpenFOAM [@weller_tensorial_1998], to the offline and online phase of each of the aforementioned DDROM algorithms. The case studies are taken from the fluid dynamics and neutronics world, being the most important physics involved in nuclear reactor physics, although the methodologies can be extended to any physics of interest.

# References
