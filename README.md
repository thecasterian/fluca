# Fluca

A Computational Fluid Dynamics (CFD) framework based on the Immersed Boundary Method (IBM) and PETSc.

## Overview

Fluca is a high-performance CFD solver designed for simulating incompressible viscous flows using the Immersed Boundary Method. Built on top of [PETSc](https://petsc.org/), it leverages parallel computing capabilities for efficient large-scale simulations.

### Features

- **Immersed Boundary Method**: Handle complex geometries without body-fitted meshes
- **Cartesian Grid Support**: Efficient structured mesh implementation
- **Navier-Stokes Solver**: Unsteady incompressible viscous flow simulation
- **Parallel Computing**: MPI-based parallelization through PETSc
- **CGNS I/O**: Standard file format support for mesh and solution data
- **Flexible Boundary Conditions**: Support for various boundary condition types
- **Extensible Architecture**: Modular design for easy addition of new solvers and mesh types

## Build

### Dependencies

- **CMake**: >= 3.20
- **C Compiler**: Compatible with PETSc
- **PETSc**: >= 3.23
- **HDF5**: Required by CGNS
- **CGNS**: Must be built with parallel I/O support enabled

### Building

Fluca uses CMake as its build system.

```bash
mkdir build
cd build
cmake ..
make
```

### CMake Options

- `PETSC_DIR`: Path to PETSc
- `HDF5_DIR`: Path to HDF5
- `CGNS_DIR`: Path to CGNS
- `FLUCA_USE_PETSC_CC`: Use the C compiler that was used to build PETSc (default: ON); if OFF, Fluca uses the compiler defined in `CMAKE_C_COMPILER`

## Output

Fluca uses the CGNS format for output files. You can visualize it using standard CFD post-processing tools such as ParaView or Tecplot.

## Development

### Code Style

The project uses `clang-format` for code formatting. The configuration file `.clang-format`, which is adopted from PETSc, is provided in the repository.

For other style guidelines, such as naming conventions or formatting not covered by `clang-format`, please refer to the [PETSc Style and Usage Guide](https://petsc.org/release/developers/style/).

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
