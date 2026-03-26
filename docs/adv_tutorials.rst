Advanced Tutorials
==================

In addition to these basic tutorials, some advanced tutorials are also available exploring more advanced capabilities of the library. These include:

1. Indirect Reconstruction of unobservable fields: two methodologies are presented to reconstruct unobservable fields from observable ones; one is based on Parameter Estimation + POD with Interpolation and the other on Gaussian Process Regression and POD (Cammi et al., 2024, NED).
2. State Estimation in Molten Salt Fast Reactors (MSFR) with failing sensors using GEIM and PBDW techniques.
3. Advanced Singular Value Decomposition (SVD) methods for large datasets, including Hierarchical SVD and Incremental SVD.

Furthermore, some examples on how to interface with other libraries are also provided (the installation of the dependencies is not included in the main installation of the library, but will be listed in the respective notebooks):

4. Combination with *pydmd* for Dynamic Mode Decomposition (DMD) analysis of time-dependent data.
5. SHallow REcurrent Decoders (SHRED) for parametric time-dependent data, implementation with *NuSHRED* and *pySHRED* codes.

.. toctree::
    :maxdepth: 1
    :caption: List of Examples:

    Indirect Reconstruction <Tutorials/Advanced/04_indirect_reconstruction.ipynb>
    Failing Sensors in MSFR <Tutorials/Advanced/05_failing_sensors.ipynb>
    Advanced SVD Methods <Tutorials/Advanced/06_advanced_svd_methods.ipynb>
    Combination with pydmd <Tutorials/Advanced/07_pydmd.ipynb>
    Implementation of SHRED <Tutorials/Advanced/08_shred.ipynb>
    Multi-Region and Parallel-Decomposed OpenFOAM Import <Tutorials/Advanced/09_multi-region.ipynb>
