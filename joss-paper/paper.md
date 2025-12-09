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
Innovative reactor technologies in the framework of Generation IV are usually characterised by harsher and more hostile environments than standard nuclear systems, for instance, due to the liquid nature of the fuel or the adoption of liquid salt or metals as coolant. These novel concepts pose more challenges in the monitoring of the system itself: depending on the specific design, high temperatures, corrosive environments, and high radiation fluence make in-core sensor deployment difficult. As a result, it is crucial to study innovative methods that combine different sources of information (namely mathematical models and measurement data, even when collected outside the core) in a quick, reliable, and efficient way. Even though, these techniques are general and can also be useful for conventional nuclear reactors or any engineering system. These methods fall into the Data-Driven Reduced Order Modelling framework, they can be very useful to learn the missing physics or the dynamics of the problem, in particular, they can be adapted to generate surrogate models able to map the out-core measurements of a simple field (e.g., neutron flux and temperature) to the dynamics of non-observable complex fields (precursors concentration and velocity).

The techniques implemented here follow the same underlying idea expressed in \autoref{fig:darom}. They all share the typical offline/online paradigm of ROM techniques: the former is computationally expensive and it is performed only once, whereas the latter is computationally cheap and allows for a quick and reliable evaluations of the state of the system by merging background model knowledge and real evaluations of quantities of interest [@maday_parameterized-background_2014].

During the offline (also called training) phase, a *high-fidelity* or Full Order Model (FOM), usually parameterised partial differential equations, is solved several times to obtain a collection of snapshots $\mathbf{u}_{FOM}\in\mathbb{R}^{\mathcal{N}_h}$, given $\mathcal{N}_h$ the dimension of the spatial mesh, which are dependent on some parameters $\boldsymbol{\mu}_n$; then, these snapshots are used to generate a reduced representation through a set of basis functions $\{\psi_n(\mathbf{x})\}$ of size $N$, in this way the degrees of freedom are decreased from $\mathcal{N}_h$ to $N$, provided that $\mathcal{N}_h>>N$. This allows for an approximation of any solution of the FOM as follows

\begin{equation}\label{eq:rb}
u(\mathbf{x};\boldsymbol{\mu}) \simeq \sum_{n=1}^N\alpha_n(\boldsymbol{\mu})\cdot \psi_n(\mathbf{x})
\end{equation}
with $\alpha_n(\boldsymbol{\mu})$ as the reduced coefficients, embedding the parametric dependence. Moreover, a reduced representation allows for the search for the optimal positions of sensors in the physical domain in a more efficient manner.

![General scheme of DDROM methods [@RMP_2024].\label{fig:darom}](../images/tie_frighter.pdf){ width=100% }

The online phase aims to obtain a quick and reliable way a solution of the FOM for an unseen parameter $\boldsymbol{\mu}^\star$, using as input a set of measurements $\mathbf{y}\in\mathbb{R}^M$. The DDROM online phase produces a novel set of reduced coordinates, $\boldsymbol{\alpha}^\star$, and then computes an improved reconstructed state $\hat{u}_{DDROM}$ through a decoding step that transforms the low-dimensional representation to the high-dimensional one.

Up to now, the techniques, reported in the following tables, have been implemented [@DDMOR_CFR;@RMP_2024]: they have been split into offline and online, including how they connect with \autoref{fig:darom}.

<!---
1. Proper Orthogonal Decomposition (POD) [@rozza_model_2020] with Projection and Interpolation [@demo_complete_2019] for the online phase
2. Generalised Empirical Interpolation Method (GEIM) [@maday_generalized_2015], either with or without Tikhonov regulation [@introini_stabilization_2023]
3. Parameterised-Background Data-Weak (PBDW) [@maday_parameterized-background_2014]
4. an Indirect Reconstruction [@introini_non-intrusive_2023] algorithm to reconstruct non-observable fields
-->

| Offline algorithm                                                           | Basis Generation | Sensor Placement |
| --------------------------------------------------------------------------- | ---------------- | ---------------- |
| Proper Orthogonal Decomposition (POD) [@rozza_model_2020]                   | X                |                  |
| SGreedy [@maday_parameterized-background_2014]                              |                  | X                |
| Generalised Empirical Interpolation Method (GEIM) [@maday_generalized_2015] | X                | X                |


| Online algorithm                                                                 | Input is parameter $\boldsymbol{\mu}$ | Input is measurement vector $\mathbf{y}$ |
| -------------------------------------------------------------------------------- | ------------------------------------- | ---------------------------------------- |
| POD Projection [@rozza_model_2020]                                               | X                                     |                                          |
| POD with Interpolation (PODI) [@demo_complete_2019]                              | X                                     |                                          |
| GEIM [@maday_generalized_2015]                                                   |                                       | X                                        |
| Tikhonov-Regularised (TR)-GEIM [@introini_stabilization_2023]                    |                                       | X                                        |
| Parameterised-Background Data-Weak (PBDW) [@maday_parameterized-background_2014] |                                       | X                                        |
| Indirect Reconstruction: parameter estimation [@introini_non-intrusive_2023]     |                                       | X                                        |

The package is organized in modules for the offline and online phases, with sub-modules for each of the algorithms mentioned above, in addition to modules for handling the data, the sensors, and the collections of functions/snapshots. The official documentation provides additional details on how the different modules and classes are related to each other and how to use them.

This package aims to become a valuable tool for other researchers, engineers, and data scientists working in various fields where multi-physics problems play an important role, and its scope of application is not only restricted to the Nuclear Engineering world, for instance it can be useful for thermo-fluid dynamics (monitoring of quantities of interest in generic energy systems) or structural mechanics problems (structural health monitoring of critical infrastructures). The data-driven nature of the implemented algorithms makes them very flexible and adaptable to different contexts. The package also includes tutorials showing how to use the library and its main features, ranging from snapshot generation in dolfinx, import and mapping from OpenFOAM [@weller_tensorial_1998], to the offline and online phase of each of the aforementioned DDROM algorithms. The case studies are taken from the fluid dynamics and neutronics world, being the most important physics involved in nuclear reactor physics, although the methodologies can be extended to any physics of interest.

# Authors contribution with [CRediT](https://credit.niso.org/)

- Stefano Riva: Conceptualization, Data curation, Formal analysis, Software, Visualization, Writing – original draft
- Carolina Introini: Conceptualization, Formal analysis, Software, Supervision, Writing – review & editing
- Antonio Cammi: Conceptualization, Project administration, Resources, Supervision, Writing – review & editing


# References
