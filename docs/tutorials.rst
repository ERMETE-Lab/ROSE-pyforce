Basic Tutorials
===============

In this section some tutorials on how to use `pyforce` will be discussed.

0. First Steps with Pyforce: learning how to use the ``FunctionsList`` class, how to store and import snapshots, how to use the ``IntegralCalculator`` class.
1. Introduction to ROM methods with SVD based methods: learning how to compress fluid dynamics data from classical flow over cylinder (generated with OpenFOAM) and create simple surrogate model for the reduced dynamics using interpolation and regression techniques.
2. Application of the EIM and GEIM methods to the parametric buoyant cavity for the temperature field.
3. Study of sensor placement methods with EIM and SGREEDY and state estimation with PBDW on a parametric reactor core problem (Riva et al., 2024, AMM).

.. toctree::
    :maxdepth: 1
    :caption: List of Tutorials:

    First Steps <Tutorials/00_first_steps.ipynb>
    SVD Methods <Tutorials/01_svd_methods.ipynb>
    (G)EIM Methods <Tutorials/02_eim_methods.ipynb>
    Sensor Placement and PBDW <Tutorials/03_sensor_placement.ipynb>