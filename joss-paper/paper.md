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
 - name: Energy Department - Nuclear Engineering Division, Nuclear Reactors Group - ERMETE Lab, Politecnico di Milano, Milan, Italy
   index: 1
date: 5 June 2024
bibliography: paper.bib

---

# Summary
*pyforce* (Python Framework for data-driven model Order Reduction of multi-physiCs problEms) is a Python library implementing Data-Driven Reduced Order Modelling (DDROM) techniques [@RMP_2024] for applications to multi-physics problems, mainly in the nuclear engineering world. These techniques have been implemented upon the dolfinx package [@BarattaEtal2023] (currently v0.6.0), part of the FEniCSx project, to handle mesh generation, integral calculation and functions storage. The package is part of the ROSE (Reduced Order modelling with data-driven techniques for multi-phySics problEms) framework, which is one of the main research topics investigated at the [ERMETE-Lab](https://github.com/ERMETE-Lab#reduced-order-modelling-with-data-driven-techniques-for-multi-physics-problems-rose-): in particular, the focus of the research activities is on mathematical algorithms aimed at reducing the complexity of multi-physics models with a focus on nuclear reactor applications, searching for optimal sensor positions and integrating experimental data to improve the knowledge on the physical systems.

# Statement of need
Innovative reactor technologies in the framework of Generation IV are usually characterised by harsher and more hostile environments than standard nuclear systems, for instance, due to the liquid nature of the fuel or the adoption of liquid salt and molten as coolant. This framework poses more challenges in the monitoring of the system itself; since placing sensors inside the reactor itself is a nearly impossible task, it is crucial to study innovative methods able to combine different sources of information, namely mathematical models and measurements data (i.e., local evaluations of quantities of interest) in a quick, reliable and efficient way. These methods fall into the Data-Driven Reduced Order Modelling framework, they can be very useful to learn the missing physics or the dynamics of the problem, in particular, they can be adapted to generate surrogate models able to map the out-core measurements of a simple field (e.g., neutron flux and temperature) to the dynamics of non-observable complex fields (precursors concentration and velocity).

![General scheme of DDROM methods [@RMP_2024].\label{fig:darom}](../images/tie_frighter.pdf){ width=100% }

The techniques implemented here follow the same underlying idea expressed in \autoref{fig:darom}. They all share the typical offline/online paradigm of ROM techniques: the former is computationally expensive and it is performed only once, whereas the latter is cheap from the computational point of view and allows to have quick and reliable evaluations of the state of the system by merging background model knowledge and real evaluations of quantities of interest [@maday_parameterized-background_2014].
During the offline (also called training) phase, a *high-fidelity* or Full Order Model (FOM), usually parameterised partial differential equations, is solved several times to obtain a collections of snapshots $\mathbf{u}_{FOM}\in\mathbb{R}^{\mathcal{N}_h}$, given $\mathcal{N}_h$ the dimension of the spatial mesh, which are dependent on some parameters $\boldsymbol{\mu}_n$


in the offline (training) phase, a dimensionality reduction process retrieves a reduced coordinate system onto which the information of the mathematical model is encoded; the sensor positioning algorithm then uses this reduced set to select the optimal location of sensors according to some optimality criterion, which depends on the adopted algorithm. In the online phase, the data assimilation process begins, retrieving a novel set of reduced variables and then computing the reconstructed state through a decoding step.

Up to now, the following techniques have been implemented [@DDMOR_CFR;@RMP_2024]:

1. Proper Orthogonal Decomposition (POD) [@rozza_model_2020] with Projection and Interpolation [@demo_complete_2019] for the Online Phase
2. Generalised Empirical Interpolation Method (GEIM) [@maday_generalized_2015], either regularised with Tikhonov [@introini_stabilization_2023] or not
3. Parameterised-Background Data-Weak (PBDW) [@maday_parameterized-background_2014]
4. an Indirect Reconstruction [@introini_non-intrusive_2023] algorithm to reconstruct non-observable fields

This package aims to become a valuable tool for other researchers, engineers, and data scientists working in various fields where multi-physics problems play an important role, and its scope of application is not only restricted to the Nuclear Engineering world. The package also includes tutorials showing how to use the library and its main features, ranging from snapshot generation in dolfinx, import and mapping from OpenFOAM [@weller_tensorial_1998], to the offline and online phase of each of the aforementioned DDROM algorithms. The case studies are taken from the fluid dynamics and neutronics world, being the most important physics involved in nuclear reactor physics, although the methodologies can be extended to any physics of interest.

# Contribution and authorship

[CRediT](https://credit.niso.org/) taxonomy has been added to clarify the roles, reported below.

- Stefano Riva: Conceptualization, Data curation, Formal analysis, Software, Visualization, Writing – original draft
- Carolina Introini: Conceptualization, Formal analysis, Software, Supervision, Writing – review & editing
- Antonio Cammi: Conceptualization, Project administration, Resources, Supervision, Writing – review & editing


# References
