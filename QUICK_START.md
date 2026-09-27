# Quick Start Guide

This guide provides a step-by-step introduction to using Fluca for CFD simulations.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Data Structures](#data-structures)
- [Basic Workflow](#basic-workflow)
- [Example: Lid-Driven Cavity Flow](#example-lid-driven-cavity-flow)
- [Command-Line Options](#command-line-options)
- [Output and Visualization](#output-and-visualization)

## Core Concepts

Fluca is built on top of PETSc and follows its object-oriented design philosophy in C. The API uses an opaque pointer pattern where users interact with objects through handles without direct access to internal data structures.

### PETSc-Style API

Fluca adopts PETSc conventions:

- **Object handles**: Objects like `Phys` and `NS` are opaque pointers
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

### DM (PETSc DMStag)

Fluca does not provide its own mesh abstraction. The computational grid is a plain PETSc `DMStag`, which you create and configure directly:

```c
DM dm;

// 2D Cartesian grid: Nx x Ny cells, non-periodic in both directions
PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                          Nx, Ny, PETSC_DECIDE, PETSC_DECIDE,
                          0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
PetscCall(DMSetFromOptions(dm));  // e.g. -stag_grid_x, -stag_grid_y
PetscCall(DMSetUp(dm));
PetscCall(DMStagSetUniformCoordinatesProduct(dm, xmin, xmax, ymin, ymax, 0., 0.));
```

Periodicity is expressed through the `DM_BOUNDARY_*` type passed at creation — there is no separate periodic flag elsewhere in the API.

### Phys (Problem Statement)

The `Phys` object states the continuous problem: the base grid, material properties, boundary conditions, and (once set up) the solution fields. Fluca currently provides one type, `PHYSLAMINAR` (isothermal laminar incompressible flow).

#### Key Functions

- **Creation and base grid**:
  ```c
  Phys phys;
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetBaseDM(phys, dm));
  ```

- **Material properties** (also settable via `-phys_density`, `-phys_viscosity`):
  ```c
  PetscCall(PhysSetDensity(phys, rho));
  PetscCall(PhysSetViscosity(phys, mu));
  ```

- **Boundary conditions**: set per face of the base DM. Faces are indexed `0`=left, `1`=right, `2`=down, `3`=up, `4`=back, `5`=front. Periodicity comes from the DMStag boundary type, not from a BC entry.
  ```c
  PhysBC wall = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL};  // fn == NULL means zero velocity
  PetscCall(PhysSetBoundaryCondition(phys, 0, wall));  // left
  PetscCall(PhysSetBoundaryCondition(phys, 1, wall));  // right
  PetscCall(PhysSetBoundaryCondition(phys, 2, wall));  // down

  PhysBC lid = {PHYS_BC_VELOCITY, LidVelocity, NULL, NULL, NULL};
  PetscCall(PhysSetBoundaryCondition(phys, 3, lid));   // up
  ```

  `PhysBC` fields:
  ```c
  typedef struct {
    PhysBCType type;      // PHYS_BC_VELOCITY (or PHYS_BC_NONE)
    PhysBCFn  *fn;        // value; NULL means zero
    void      *ctx;
    PhysBCFn  *fn_dot;    // time derivative; NULL means a finite difference of fn
    void      *fn_dot_ctx;
  } PhysBC;
  ```
  where the callback signature is:
  ```c
  PetscErrorCode PhysBCFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx);
  ```

- **Configuration and setup**:
  ```c
  PetscCall(PhysSetFromOptions(phys));  // apply command-line options
  ```
  `PhysSetUp()` (called implicitly by `NSSetUp()`) declares the solution fields and lays out one DMStag holding all of them.

- **Solution fields**: `PHYS_FIELD_VELOCITY` (cell-centered velocity), `PHYS_FIELD_FACE_VELOCITY` (face-normal velocity), `PHYS_FIELD_PRESSURE` (cell-centered pressure). Field metadata and index sets are queried with `PhysGetField()` / `PhysGetFieldIS()` after setup.

- **Cleanup**:
  ```c
  PetscCall(PhysDestroy(&phys));
  ```

### NS (Navier-Stokes Solver)

The `NS` object solves the problem stated by a `Phys`. It manages time integration and the solution vector.

#### Key Functions

- **Creation and type**:
  ```c
  NS ns;
  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSCNLINEAR));  // Linearized Crank-Nicolson
  ```

- **Associating with a Phys**:
  ```c
  PetscCall(NSSetPhys(ns, phys));
  ```

- **Time stepping** (also settable via `-ns_time_step_size`, `-ns_max_steps`):
  ```c
  PetscCall(NSSetTimeStepSize(ns, dt));
  PetscCall(NSSetMaxSteps(ns, nsteps));
  PetscCall(NSSetMaxTime(ns, tmax));  // optional additional stopping condition
  ```

- **Configuration and setup**:
  ```c
  PetscCall(NSSetFromOptions(ns));  // apply command-line options
  PetscCall(NSSetUp(ns));           // builds the Phys solution DM if not already set up
  ```

- **Solving**:
  ```c
  PetscCall(NSSolve(ns));  // advances until a converged reason is reached (max steps / max time)
  ```

- **Accessing the solution**:
  ```c
  Vec sol, velocity;
  PetscCall(NSGetSolution(ns, &sol));                                    // the full monolithic solution vector
  PetscCall(NSGetSolutionSubVector(ns, PHYS_FIELD_VELOCITY, &velocity)); // borrowed sub-vector for one field
  // Use velocity...
  PetscCall(NSRestoreSolutionSubVector(ns, PHYS_FIELD_VELOCITY, &velocity));
  ```

- **Cleanup**:
  ```c
  PetscCall(NSDestroy(&ns));
  ```

## Basic Workflow

A typical Fluca simulation follows this workflow:

1. Initialize Fluca
2. Create and configure the base DMStag grid
3. Create the `Phys` (base DM, density, viscosity, boundary conditions)
4. Create and configure the `NS` solver on that `Phys`
5. Set the initial condition on the solution vector
6. Solve
7. Clean up

## Example: Lid-Driven Cavity Flow

The lid-driven cavity is a classic benchmark problem in CFD. This example demonstrates the basic API usage; it mirrors `fluca/tutorials/ns/ex1.c`.

```c
#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "2D lid-driven cavity flow with NS\n";

/* u = 1 on the lid (the up face, where this BC is attached) */
static PetscErrorCode LidVelocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = comp == 0 ? 1. : 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm;
  Phys      phys;
  NS        ns;
  Vec       sol;
  PhysBC    wall = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL}, lid = {PHYS_BC_VELOCITY, LidVelocity, NULL, NULL, NULL};
  PetscReal Re = 100.;
  PetscInt  f;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Re", &Re, NULL));

  // 1. Create the base DMStag grid (256x256 cells in default)
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 256, 256,
                            PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  // 2. Create and configure the Phys (density 1, viscosity 1/Re)
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysSetDensity(phys, 1.));
  PetscCall(PhysSetViscosity(phys, 1. / Re));
  for (f = 0; f < 3; f++) PetscCall(PhysSetBoundaryCondition(phys, f, wall)); // left, right, down
  PetscCall(PhysSetBoundaryCondition(phys, 3, lid));                          // up (moving lid)
  PetscCall(PhysSetFromOptions(phys));

  // 3. Create and configure the Navier-Stokes solver
  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSCNLINEAR));      // Linearized Crank-Nicolson
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetFromOptions(ns));           // e.g. -ns_time_step_size, -ns_max_steps
  PetscCall(NSSetUp(ns));

  // 4. Set the initial condition and solve
  PetscCall(NSGetSolution(ns, &sol));
  PetscCall(VecZeroEntries(sol));
  PetscCall(NSSolve(ns));

  // 5. Clean up
  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}
```

Run it with:

```bash
./ex1 -stag_grid_x 64 -stag_grid_y 64 -Re 100 -ns_time_step_size 0.002 -ns_max_steps 100
```

See `fluca/tutorials/ns/ex1.c` (lid-driven cavity) and `fluca/tutorials/ns/ex2.c` (Taylor-Green vortex, with an exact-solution error check) for complete, runnable examples.

## Output and Visualization

`NSViewSolution(ns, viewer)` is a plain `VecView` of the solution vector on the Phys solution DM — it accepts any `PetscViewer` (e.g. ASCII, binary):

```c
PetscViewer viewer;

PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "solution.txt", &viewer));
PetscCall(NSViewSolution(ns, viewer));
PetscCall(PetscViewerDestroy(&viewer));
```

There is currently no dedicated CGNS solution I/O or a solution-loading entry point on `NS` — those were removed along with the standalone application target. Use the fields of the solution vector directly (`NSGetSolutionSubVector`) for post-processing within your own program.

## Next Steps

- Explore the full source code in `fluca/tutorials/ns/` and `fluca/tests/`
- Read the [THEORY_GUIDE.md](THEORY_GUIDE.md) for detailed mathematical background
- Refer to [PETSc documentation](https://petsc.org/release/manual/) for advanced solver options
- Check the [README.md](README.md) for build instructions and dependencies
