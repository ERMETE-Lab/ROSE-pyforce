Laminar Flow Past a Cylinder using OpenFOAM-v2012
=============================================================

**Aim of this tutorial:** learn how to import transient snapshots from OpenFOAM (version 2012 from com-version) using *pyvista* package, generate the POD modes from the snapshots for the Flow Over Cylinder problem using an accelerated version of the algorithm and create a reduced model using Gaussian Process Regression (GPR).

The Flow Over Cylinder problem is a well-known benchmark in fluid dynamics, where a cylinder is placed in a flow field, leading to the formation of vortices and complex flow patterns. The problem is solved using OpenFOAM, a popular open-source CFD software (*pimpleFoam* solver in this case). The governing equations are the Navier Stokes equations:

.. math::
    \left\{
        \begin{array}{ll}
        \nabla\cdot \mathbf{u} = 0 & \mathbf{x}\in\Omega\\
        \frac{\partial \mathbf{u}}{\partial t} + \left(\mathbf{u}\cdot \nabla\right)\mathbf{u} -\nu \Delta \mathbf{u} + \nabla p = 0 & \mathbf{x}\in\Omega\\
        \mathbf{u}=\mathbf{u}_{in},\;\; \frac{\partial p}{\partial \mathbf{n}} = 0 & \mathbf{x}\in\Gamma_{in} \\
        \mathbf{u}=\mathbf{0},\;\; \frac{\partial p}{\partial \mathbf{n}} = 0 & \mathbf{x}\in\Gamma_{w} \\
        \frac{\partial \mathbf{u}}{\partial \mathbf{n}}=0,\;\; p = 0 & \mathbf{x}\in\Gamma_{out}
        \end{array}
    \right.

given :math:`\Omega` as the domain and :math:`\partial\Omega` as its boundary, composed by :math:`\partial\Omega = \Gamma_{in}\cup\Gamma_{w}\cup\Gamma_{out}` where :math:`\Gamma_{in}` is the inlet boundary, :math:`\Gamma_{w}` is the wall boundary and :math:`\Gamma_{out}` is the outlet.

.. toctree::
    :maxdepth: 3
    :caption: Steps:

    Import Snapshots from OF <04_FlowOverCylinder_OF2012/01_import_snaps.ipynb>
    Generation and testing of the surrogate mode: <04_FlowOverCylinder_OF2012/02_POD_GPR.ipynb>