Unsteady Laminar Navier Stokes - DFG2 benchmark
=========================================================

The DFG2 benchmark is taken from the `FeatFlow Repository <https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html>`_. 
It's 2D version of a Flow Over Cylinder Problem, able to observe Von-Karman vortex shedding.

.. image:: ../images/Tutorials/dfg2_ns_benchmark.png
  :width: 750
  :alt: dfg2_ns_benchmark
  :align: center

The governing equations are the Navier Stokes equations:

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

    Generation of the snapshots <01_LaminarNS/01_generateFlowData.ipynb>
    Offline Phase <01_LaminarNS/02_offline_POD.ipynb>
    Online Phase: POD-I <01_LaminarNS/03_online_POD-I.ipynb>
