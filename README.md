# Fluca

A computational fluid dynamics (CFD) framework for incompressible viscous flows on Cartesian grids, built on [PETSc](https://petsc.org/).

## Overview

Fluca solves the incompressible Navier-Stokes equations using finite differences on collocated Cartesian grids with pressure stabilization. Time integration uses the Segregated Runge-Kutta (SRK) method, which decouples the velocity-pressure system into Helmholtz and Poisson solves at each stage.

### Features

- **Collocated grid**: Cell-centered velocity and pressure with pressure stabilization
- **FlucaFD operators**: Composable finite difference operators (derivatives, compositions, scaling, sums, TVD schemes)
- **Segregated Runge-Kutta**: IMEX time integration with 12 selectable schemes (Bakhvalov, 2025)
- **Physics models**: Modular `Phys` abstraction with incompressible Navier-Stokes (`PhysINS`) subtype
- **Parallel computing**: MPI-based parallelization through PETSc
- **CGNS I/O**: Standard file format support for solution data

## Build

### Dependencies

- **CMake**: >= 3.20
- **C Compiler**: Compatible with PETSc
- **PETSc**: >= 3.23
- **HDF5**: Required by CGNS
- **CGNS**: Must be built with parallel I/O support enabled

### Building

```bash
cmake -B build
cmake --build build
```

### Running Tests

```bash
ctest --test-dir build -R tests_       # Unit tests (fast)
ctest --test-dir build -R tutorials_   # Tutorial tests (slow)
```

### CMake Options

- `PETSC_DIR`: Path to PETSc
- `HDF5_DIR`: Path to HDF5
- `CGNS_DIR`: Path to CGNS
- `FLUCA_USE_PETSC_CC`: Use the C compiler that was used to build PETSc (default: ON); if OFF, Fluca uses the compiler defined in `CMAKE_C_COMPILER`

## Getting Started

See [QUICK_START.md](docs/QUICK_START.md) for a step-by-step guide to building simulations with Fluca.

## Theory Guide

See [THEORY_GUIDE.md](docs/THEORY_GUIDE.md) for the mathematical formulation and numerical methods.

## Output

Fluca uses the CGNS format for output files. You can visualize results using standard CFD post-processing tools such as ParaView or Tecplot.

## Development

### Code Style

The project uses `clang-format` for code formatting. The configuration file `.clang-format`, adopted from PETSc, is provided in the repository.

For other style guidelines, such as naming conventions or formatting not covered by `clang-format`, refer to the [PETSc Style and Usage Guide](https://petsc.org/release/developers/style/).

### Pre-commit Hooks

Pre-commit hooks are configured in `.pre-commit-config.yaml`. Install them with:

```bash
pip install pre-commit
pre-commit install
```

## License

This project is licensed under the BSD 3-Clause License. See the [LICENSE](LICENSE) file for details.

## References

- PETSc: [https://petsc.org/](https://petsc.org/)
- CGNS: [https://cgns.github.io/](https://cgns.github.io/)
- S. Bakhvalov, Segregated Runge-Kutta methods for the incompressible Navier-Stokes equations, [arXiv:2506.09519](https://arxiv.org/abs/2506.09519) (2025)
