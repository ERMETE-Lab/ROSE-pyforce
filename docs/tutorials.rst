Tutorials
============

In this section some tutorials on how to use pyforce will be discussed.

The following case studies are discussed:

1. **Flow over Cylinder** (solved with FEniCSx): DFG2 benchmark

    - *Offline*: Generation of parametric snapshots.
    - *Offline*: Creation of the reduced space using POD for vector fields and plotting the modes.
    - *Online*:  POD with Interpolation adopting linear and RBF interpolation.

2. **Stationary Multi-group Diffusion equation**: ANL11-A2 reactor from the Argonne Code Center - Supplement 2 at https://www.osti.gov/biblio/12030251. The following topics will be discussed:

    - *Offline*: Generation of parametric snapshots using *FunctionsList* class and later export.
    - *Offline*: Creation of the reduced space using the Proper Orthogonal Decomposition (POD).
    - *Offline*: Generalised Empirical Interpolation Method (GEIM) to generate basis functions and place sensors.
    - *Offline*: Sensor Placement with the SGREEDY algorithm using the POD basis.
    - *Online*:  Effect of random noise onto the GEIM reconstruction, stabilisation with TR-GEIM.
    - *Online*:  Direct State estimation Parameterised-Background Data-Weak formulation (PBDW), considering noisy data.

3. **Buoyant Cavity** (solved with OpenFOAM-6, taken from `ROSE-ROM4FOAM <https://ermete-lab.github.io/ROSE-ROM4FOAM/Tutorials/BuoyantCavity/problem.html>`_)

    - *Offline*: Import from OpenFOAM and plotting using *pyvista*.
    - *Offline*: Creation of the reduced space using the POD and GEIM.
    - *Online*:  Indirect Reconstruction (PE+POD-I), considering noisy data.

.. toctree::
    :maxdepth: 1
    :caption: List of Tutorials:

    Unsteady Laminar Navier-Stokes - DFG2 benchmark <Tutorials/01_DFG2_benchmark.rst>
    MultiGroup Neutron Diffusion - ANL11-A2 benchmark <Tutorials/02_ANL11-A2_stationary.rst>
    Steady Buoyant Navier-Stokes - Differentially Heated Cavity <Tutorials/03_buoyant_cavity.rst>
