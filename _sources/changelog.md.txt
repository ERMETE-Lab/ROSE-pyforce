# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-xx-yy
*The "Solver-Agnostic" Release*

### Added
- **PyVista Backend:** Complete integration with `pyvista` for mesh handling, integral calculations and visualization.
- **Numpy-native:** All algorithms now accept standard `numpy` arrays, removing the need for FEM objects.
- **Universal Input:** Support for importing results from OpenFOAM, Ansys, and any VTK-supported solver.
- **New Tutorials:** Added tutorials for non-FEniCS workflows (e.g., fluid dynamics with external data).

### Changed
- **Removed Dependency:** Dropped `dolfinx` and `mpi4py` as core dependencies.
- **Refactor:** `POD`, `EIM`, `SGreedy` and other classes now operate on matrix inputs rather than FEM functions.
- **Docs:** Completely rewritten documentation to reflect the new architecture.
- Improved installation speed and compatibility with Linux/macOS.

---

## [0.1.3] - JOSS paper published 2026-01-09
*The "JOSS Submission" Release*

### Added
- Implementation of POD, EIM, GEIM, and PBDW algorithms based on `dolfinx`.
- Support for distributed computing via MPI.
- Original application cases for Nuclear Engineering (Circulating Fuel Reactor).

### Reference
- The original version corresponds to the implementation described in the JOSS paper: *Riva et al., "pyforce: Python Framework for data-driven model Order Reduction of multi-physiCs problems"*
