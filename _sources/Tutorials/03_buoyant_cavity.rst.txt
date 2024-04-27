Steady Buoyant Navier-Stokes for Differentially Heated Cavity
=============================================================

The Differentially Heated Cavity is taken from the `ROSE-ROM4FOAM tutorial <https://ermete-lab.github.io/ROSE-ROM4FOAM/Tutorials/BuoyantCavity/problem.html>`_.
It's 2D Cavity with velocity imposed at the left and right boundary, along with :cite:p:`Saha2018`.

.. image:: ../images/Tutorials/two_sided_lid_driven_cavity.png
  :width: 350
  :alt: cavity
  :align: center

The governing equations are the Navier-Stokes with energy equation under the Boussinesq approximation

.. math::
    \left\{ \begin{aligned} 
    \nabla \cdot \mathbf{u}&=0 \quad &\text{ in } \Omega\\
    (\mathbf{u} \cdot \nabla)\mathbf{u} - \nu \Delta \mathbf{u}+ \nabla p - \mathbf{g}\cdot\beta(T-T_\infty) &=0 & \text{ in } \Omega \\
    \mathbf{u} \cdot \nabla T - \alpha \Delta T&= 0\quad& \text{ in } \Omega 
    \end{aligned} \right.
    
given :math:`\Omega` as the domain.

.. toctree::
    :maxdepth: 3
    :caption: Steps:

    Generation of the snapshots <03_BuoyantCavity_OF6/01_import_OFsnaps.ipynb>
    Offline Phase: POD <03_BuoyantCavity_OF6/02a_offline_POD.ipynb>
    Offline Phase: GEIM <03_BuoyantCavity_OF6/02b_offline_GEIM.ipynb>
    Offline Phase: generation of the maps <03_BuoyantCavity_OF6/02c_offline_maps.ipynb>
    Online Phase: Indirect Reconstruction <03_BuoyantCavity_OF6/03_online_IR.ipynb>