# Quick Start Guide

This guide provides a step-by-step introduction to using Fluca for CFD simulations.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Data Structures](#data-structures)
- [Basic Workflow](#basic-workflow)
- [Example 1: Lid-Driven Cavity Flow](#example-1-lid-driven-cavity-flow)
- [Example 2: Taylor-Green Vortex](#example-2-taylor-green-vortex)
- [Command-Line Options](#command-line-options)
- [Output and Visualization](#output-and-visualization)

## Core Concepts

Fluca is built on top of PETSc and follows its object-oriented design philosophy in C. The API uses an opaque pointer pattern where users interact with objects through handles without direct access to internal data structures.

### PETSc-Style API

Fluca adopts PETSc conventions:

- **Object handles**: Objects like `Mesh` and `NS` are opaque pointers
- **Create-SetUp-Use-Destroy pattern**: Objects are created, configured, used, then destroyed
- **Error handling**: Functions return `PetscErrorCode`; use `PetscCall()` macro for error checking
- **MPI parallelism**: Built-in support for parallel computing via MPI communicators

### Initialization and Finalization

Every Fluca program must initialize and finalize the library:

```c
int main(int argc, char **argv)
{
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  // Your simulation code here

  PetscCall(FlucaFinalize());
  return 0;
}
```

## Data Structures

### Mesh

The `Mesh` object represents the computational grid. Fluca currently supports Cartesian grids through the `MESHCART` type.

#### Key Functions

- **Creation**:
  ```c
  Mesh mesh;
  // Create 2D Cartesian mesh: nx×ny cells, with boundary types and parallel distribution
  PetscCall(MeshCartCreate2d(PETSC_COMM_WORLD,
                             MESHCART_BOUNDARY_NONE,      // X-boundary type
                             MESHCART_BOUNDARY_NONE,      // Y-boundary type
                             M, N,                        // Grid dimensions
                             PETSC_DECIDE, PETSC_DECIDE,  // Number of processors (auto)
                             NULL, NULL,                  // Parallel distribution (NULL = uniform)
                             &mesh));
  ```

- **Boundary Types**:
  - `MESHCART_BOUNDARY_NONE`: Non-periodic boundary
  - `MESHCART_BOUNDARY_PERIODIC`: Periodic boundary

- **Configuration and Setup**:
  ```c
  PetscCall(MeshSetFromOptions(mesh));  // Apply command-line options
  PetscCall(MeshSetUp(mesh));           // Finalize mesh configuration
  ```

- **Setting Coordinates**:
  ```c
  // Set uniform coordinates for the domain
  PetscCall(MeshCartSetUniformCoordinates(mesh,
                                          xmin, xmax,  // X range
                                          ymin, ymax,  // Y range
                                          zmin, zmax));// Z range (0,0 for 2D)
  ```

- **Boundary Indexing**:
  ```c
  PetscInt ileft, iright, idown, iup;
  PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_LEFT, &ileft));
  PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_RIGHT, &iright));
  PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_DOWN, &idown));
  PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_UP, &iup));
  ```

- **Accessing PETSc DM Objects**:
  ```c
  DM dm;
  PetscCall(MeshGetDM(mesh, MESH_DM_SCALAR, &dm));  // For scalar fields
  PetscCall(MeshGetDM(mesh, MESH_DM_VECTOR, &dm));  // For vector fields
  ```

- **Cleanup**:
  ```c
  PetscCall(MeshDestroy(&mesh));
  ```

### NS (Navier-Stokes Solver)

The `NS` object represents the Navier-Stokes solver. It manages the governing equations, boundary conditions, and solution vectors.

#### Key Functions

- **Creation**:
  ```c
  NS ns;
  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSFSM));  // Fractional Step Method
  ```

- **Associating with Mesh**:
  ```c
  PetscCall(NSSetMesh(ns, mesh));
  ```

- **Physical Parameters**:
  ```c
  PetscCall(NSSetDensity(ns, rho));      // Fluid density
  PetscCall(NSSetViscosity(ns, mu));     // Dynamic viscosity
  PetscCall(NSSetTimeStepSize(ns, dt));  // Time step size
  ```

- **Boundary Conditions**:
  ```c
  NSBoundaryCondition bc = {
    .type         = NS_BC_VELOCITY,  // or NS_BC_PERIODIC
    .velocity     = velocity_func,   // User-defined callback
    .ctx_velocity = &user_context,   // Optional context pointer
  };
  PetscCall(NSSetBoundaryCondition(ns, boundary_index, bc));
  ```

  Boundary condition types:
  - `NS_BC_VELOCITY`: Dirichlet velocity boundary condition
  - `NS_BC_PERIODIC`: Periodic boundary condition

- **Configuration and Setup**:
  ```c
  PetscCall(NSSetFromOptions(ns));  // Apply command-line options
  PetscCall(NSSetUp(ns));           // Finalize solver configuration
  ```

- **Solving**:
  ```c
  PetscInt nsteps = 100;
  PetscCall(NSSolve(ns, nsteps));  // Advance solution by nsteps
  ```

- **Accessing Solution Vectors**:
  ```c
  Vec velocity, pressure, face_velocity;
  PetscCall(NSGetSolutionSubVector(ns, NS_FIELD_VELOCITY, &velocity));
  PetscCall(NSGetSolutionSubVector(ns, NS_FIELD_PRESSURE, &pressure));
  PetscCall(NSGetSolutionSubVector(ns, NS_FIELD_FACE_NORMAL_VELOCITY, &face_velocity));

  // Use the vectors...

  PetscCall(NSRestoreSolutionSubVector(ns, NS_FIELD_VELOCITY, &velocity));
  PetscCall(NSRestoreSolutionSubVector(ns, NS_FIELD_PRESSURE, &pressure));
  PetscCall(NSRestoreSolutionSubVector(ns, NS_FIELD_FACE_NORMAL_VELOCITY, &face_velocity));
  ```

- **Cleanup**:
  ```c
  PetscCall(NSDestroy(&ns));
  ```

### NSBoundaryCondition

A struct that defines boundary conditions:

```c
typedef struct {
  NSBoundaryConditionType type;         // Boundary condition type
  PetscErrorCode (*velocity)(PetscInt dim, PetscReal t, const PetscReal x[],
                             PetscScalar val[], void *ctx);
  void *ctx_velocity;                   // Optional context for velocity function
} NSBoundaryCondition;
```

The velocity callback function signature:
- `dim`: Spatial dimension (2 or 3)
- `t`: Current time
- `x[]`: Spatial coordinates
- `val[]`: Output velocity components (to be filled)
- `ctx`: User-provided context pointer

### PetscViewer

Fluca provides a custom CGNS-based viewer:

```c
PetscViewer viewer;
PetscCall(PetscViewerFlucaCGNSOpen(comm, filename, mode, &viewer));
// mode: FILE_MODE_READ, FILE_MODE_WRITE
```

## Basic Workflow

A typical Fluca simulation follows this workflow:

1. Initialize Fluca
2. Create and set up the mesh
3. Create and configure the Navier-Stokes solver
4. Set initial conditions and boundary conditions
5. Solve the problem
6. Clean up

## Example: Lid-Driven Cavity Flow

The lid-driven cavity is a classic benchmark problem in CFD. This example demonstrates the basic API usage.

### Code Structure

```c
#include <flucameshcart.h>
#include <flucans.h>
#include <flucaviewer.h>
#include <math.h>

const char *help = "lid-driven cavity flow test\n";

int main(int argc, char **argv)
{
  Mesh        mesh;
  NS          ns;
  PetscViewer viewer;

  // 1. Initialize Fluca
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  // 2. Create mesh (8x8 Cartesian grid in default)
  PetscCall(MeshCartCreate2d(PETSC_COMM_WORLD,
                             MESHCART_BOUNDARY_NONE,
                             MESHCART_BOUNDARY_NONE,
                             8, 8,
                             PETSC_DECIDE, PETSC_DECIDE,
                             NULL, NULL, &mesh));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));

  // 3. Create and configure Navier-Stokes solver
  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSFSM));           // Fractional step method
  PetscCall(NSSetMesh(ns, mesh));
  PetscCall(NSSetDensity(ns, 400.0));        // Set density
  PetscCall(NSSetViscosity(ns, 1.0));        // Set viscosity
  PetscCall(NSSetTimeStepSize(ns, 0.002));   // Set time step

  // 4. Set boundary conditions
  {
    PetscInt            ileftb, irightb, idownb, iupb;
    NSBoundaryCondition bcwall = {
      .type         = NS_BC_VELOCITY,
      .velocity     = wall_velocity,         // User-defined function
      .ctx_velocity = NULL,
    };
    NSBoundaryCondition bcmovingwall = {
      .type         = NS_BC_VELOCITY,
      .velocity     = moving_wall_velocity,  // User-defined function
      .ctx_velocity = NULL,
    };

    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_LEFT, &ileftb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_RIGHT, &irightb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_DOWN, &idownb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_UP, &iupb));

    PetscCall(NSSetBoundaryCondition(ns, ileftb, bcwall));
    PetscCall(NSSetBoundaryCondition(ns, irightb, bcwall));
    PetscCall(NSSetBoundaryCondition(ns, idownb, bcwall));
    PetscCall(NSSetBoundaryCondition(ns, iupb, bcmovingwall));
  }

  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));

  // 5. Solve for specified number of steps
  PetscCall(NSSolve(ns, 10));

  // 6. Clean up
  PetscCall(MeshDestroy(&mesh));
  PetscCall(NSDestroy(&ns));
  PetscCall(FlucaFinalize());
}
```

### Boundary Condition Functions

Boundary conditions are specified through callback functions:

```c
// Stationary wall (u = 0, v = 0)
static PetscErrorCode wall_velocity(PetscInt dim, PetscReal t,
                                     const PetscReal x[],
                                     PetscScalar val[], void *ctx)
{
  PetscInt i;
  for (i = 0; i < dim; ++i) val[i] = 0.0;
  return PETSC_SUCCESS;
}

// Moving wall (u = 1.0, v = 0)
static PetscErrorCode moving_wall_velocity(PetscInt dim, PetscReal t,
                                            const PetscReal x[],
                                            PetscScalar val[], void *ctx)
{
  PetscInt i;
  for (i = 0; i < dim; ++i) val[i] = (i == 0) ? 1.0 : 0.0;
  return PETSC_SUCCESS;
}
```

### Running the Simulation

```bash
# Run with default parameters
./cavity

# Run with custom grid size (via command-line options)
./cavity -cart_grid_x 64 -cart_grid_y 64

# Save solution in a CGNS file
./cavity -ns_view_solution cgns:cavity.cgns
```

## Output and Visualization

### CGNS Output

Fluca outputs solution data in CGNS format, which is widely supported by CFD post-processing tools.

### Visualizing with ParaView

1. Open ParaView
2. File → Open → Select the `.cgns` file
3. Click "Apply" in the Properties panel
4. Select variables to visualize (velocity, pressure, etc.)
5. Use filters for streamlines, contours, etc.

### Programmatic Output

You can control output programmatically:

```c
PetscViewer viewer;

// Open viewer for writing
PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD,
                                   "output.cgns",
                                   FILE_MODE_WRITE,
                                   &viewer));

// Write solution
PetscCall(NSViewSolution(ns, viewer));

// Close viewer
PetscCall(PetscViewerDestroy(&viewer));
```

### Loading Previous Solutions

To restart from a saved solution:

```c
PetscViewer viewer;

// Open viewer for reading
PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD,
                                   "input.cgns",
                                   FILE_MODE_READ,
                                   &viewer));

// Load solution
PetscCall(NSLoadSolution(ns, viewer));

// Close viewer
PetscCall(PetscViewerDestroy(&viewer));

// Continue simulation
PetscCall(NSSolve(ns, additional_steps));
```

## Next Steps

- Explore the full source code in tests
- Read the [THEORY_GUIDE.md](THEORY_GUIDE.md) for detailed mathematical background
- Refer to [PETSc documentation](https://petsc.org/release/manual/) for advanced solver options
- Check the [README.md](README.md) for build instructions and dependencies
