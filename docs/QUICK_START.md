# Quick Start Guide

This guide introduces the main Fluca classes and walks through a complete simulation.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Data Structures](#data-structures)
- [Example: Taylor-Green Vortex](#example-taylor-green-vortex)
- [Command-Line Options](#command-line-options)
- [Output and Visualization](#output-and-visualization)

## Core Concepts

Fluca is built on PETSc and follows its object-oriented C conventions:

- **Opaque handles**: Objects like `Phys`, `Seg`, and `FlucaFD` are pointers to private structs
- **Create-SetUp-Use-Destroy**: Objects are created, configured, set up, used, then destroyed in reverse order
- **Error handling**: Functions return `PetscErrorCode`; wrap every call with `PetscCall()`
- **MPI parallelism**: Objects take an MPI communicator at creation time

Every Fluca program must initialize and finalize the library:

```c
#include <flucasys.h>

int main(int argc, char **argv)
{
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  /* ... */
  PetscCall(FlucaFinalize());
}
```

## Data Structures

### Mesh / MeshCart

The `Mesh` object represents the computational grid. `MESHCART` provides Cartesian grids.

```c
#include <flucameshcart.h>

Mesh mesh;
PetscCall(MeshCartCreate2d(PETSC_COMM_WORLD,
                           MESHCART_BOUNDARY_PERIODIC,   /* x boundary */
                           MESHCART_BOUNDARY_PERIODIC,   /* y boundary */
                           64, 64,                       /* grid cells */
                           PETSC_DECIDE, PETSC_DECIDE,   /* MPI ranks */
                           NULL, NULL, &mesh));
PetscCall(MeshSetFromOptions(mesh));
PetscCall(MeshSetUp(mesh));
PetscCall(MeshCartSetUniformCoordinates(mesh, 0., 1., 0., 1., 0., 0.));

/* Access PETSc DM objects */
DM dm;
PetscCall(MeshGetDM(mesh, MESH_DM_STAG_SCALAR, &dm));

PetscCall(MeshDestroy(&mesh));
```

Boundary types:
- `MESHCART_BOUNDARY_NONE`: Non-periodic (wall) boundary
- `MESHCART_BOUNDARY_PERIODIC`: Periodic boundary

### FlucaFD

`FlucaFD` is a polymorphic finite difference operator on PETSc `DMStag`. Operators are composable: you build complex discretizations by combining simple primitives.

**Subtypes:**

| Type | Description | Constructor |
|------|-------------|-------------|
| `derivative` | Finite difference derivative | `FlucaFDDerivativeCreate` |
| `composition` | Composition of two operators (outer . inner) | `FlucaFDCompositionCreate` |
| `scale` | Multiply by constant or spatially-varying field | `FlucaFDScaleCreateConstant`, `FlucaFDScaleCreateVector` |
| `sum` | Sum of multiple operators | `FlucaFDSumCreate` |
| `secondordertvd` | TVD convection with flux limiters | `FlucaFDSecondOrderTVDCreate` |

**Example**: Build a second derivative operator (Laplacian in one direction):

```c
#include <flucafd.h>

/* d/dx: element -> left face */
FlucaFD d_dx_ef;
PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 1,
                                  DMSTAG_ELEMENT, 0,   /* input: element, comp 0 */
                                  DMSTAG_LEFT, 0,      /* output: left face, comp 0 */
                                  &d_dx_ef));
PetscCall(FlucaFDSetUp(d_dx_ef));

/* d/dx: left face -> element */
FlucaFD d_dx_fe;
PetscCall(FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 1,
                                  DMSTAG_LEFT, 0,
                                  DMSTAG_ELEMENT, 0,
                                  &d_dx_fe));
PetscCall(FlucaFDSetUp(d_dx_fe));

/* d^2/dx^2 = d_dx_fe . d_dx_ef */
FlucaFD d2_dx2;
PetscCall(FlucaFDCompositionCreate(d_dx_fe, d_dx_ef, &d2_dx2));
PetscCall(FlucaFDSetUp(d2_dx2));

/* Apply: output = d^2/dx^2 (input) */
PetscCall(FlucaFDApply(d2_dx2, 0., dm_in, dm_out, vec_in, vec_out));

/* Or assemble into a matrix */
PetscCall(FlucaFDGetOperator(d2_dx2, dm_in, dm_out, mat));

PetscCall(FlucaFDDestroy(&d2_dx2));
PetscCall(FlucaFDDestroy(&d_dx_fe));
PetscCall(FlucaFDDestroy(&d_dx_ef));
```

TVD flux limiters (selectable via `-flucafd_limiter <name>`): `superbee` (default), `minmod`, `mc`, `vanleer`, `vanalbada`, `barthjesperson`, `venkatakrishnan`, `koren`, `upwind`, `sou`, `quick`.

### Phys / PhysINS

`Phys` is the physics model abstraction. The `PHYSINS` subtype provides incompressible Navier-Stokes, constructing all the FlucaFD operators needed for spatial discretization.

```c
#include <flucaphys.h>

Phys phys;
PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
PetscCall(PhysSetType(phys, PHYSINS));
PetscCall(PhysSetBaseDM(phys, dm));
PetscCall(PhysINSSetDensity(phys, 1.0));
PetscCall(PhysINSSetViscosity(phys, 0.01));
PetscCall(PhysSetFromOptions(phys));
PetscCall(PhysSetUp(phys));

/* Get the solution DM (may differ from base DM in DOF layout) */
DM sol_dm;
PetscCall(PhysGetSolutionDM(phys, &sol_dm));

PetscCall(PhysDestroy(&phys));
```

**Boundary conditions** (non-periodic boundaries):

```c
/* Velocity BC callback */
static PetscErrorCode WallBC(PetscInt dim, PetscReal t, const PetscReal x[],
                             PetscInt comp, PetscScalar *val, void *ctx)
{
  *val = 0.;
  return PETSC_SUCCESS;
}

PhysINSBC bc = {.type = PHYS_INS_BC_VELOCITY, .fn = WallBC, .ctx = NULL};
PetscCall(PhysINSSetBoundaryCondition(phys, boundary_index, bc));
```

**Body forces**:

```c
static PetscErrorCode BodyForce(PetscInt dim, PetscReal t, const PetscReal x[],
                                PetscScalar f[], void *ctx)
{
  f[0] = 0.;  /* x-component */
  f[1] = 0.;  /* y-component */
  return PETSC_SUCCESS;
}

PetscCall(PhysSetBodyForce(phys, BodyForce, NULL));
```

### Seg / SegSRK

`Seg` is the segregated time integrator. The `SEGSRK` subtype implements the Segregated Runge-Kutta method (Bakhvalov, 2025), which decouples the velocity-pressure system into a Helmholtz solve (viscous diffusion) and a Poisson solve (pressure correction) at each stage.

```c
#include <flucaseg.h>

Seg seg;
PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
PetscCall(SegSetType(seg, SEGSRK));

/* Wire FlucaFD operators from Phys to Seg */
PetscCall(PhysSetUpSeg(phys, seg));

/* Set solution vector and time parameters */
PetscCall(SegSetSolution(seg, Y));
PetscCall(SegSetTimeStepSize(seg, 0.01));
PetscCall(SegSetMaxTime(seg, 1.0));
PetscCall(SegSetFromOptions(seg));
PetscCall(SegSetUp(seg));

/* Run */
PetscCall(SegSolve(seg));         /* full time loop */
/* or: PetscCall(SegStep(seg));   single time step */

PetscCall(SegDestroy(&seg));
```

**IMEX RK schemes** (selectable via `-seg_srk_type <name>`):

| Name | Order | Stages | Recommended |
|------|-------|--------|-------------|
| `ars343` | 3 | 4 | Default; best ARS-type scheme |
| `ark436l2sa` | 4 | 6 | Best overall performance |
| `bhr553` | 3 | 5 | No pressure order reduction |
| `ars111` | 1 | 2 | |
| `ars121` | 1 | 2 | |
| `ars222` | 2 | 3 | |
| `ars232` | 2 | 3 | |
| `ars443` | 3 | 5 | |
| `ark324l2sa` | 3 | 4 | |
| `ark548l2sa` | 5 | 8 | |
| `mark324l2sa` | 3 | 4 | |
| `mars343` | 3 | 4 | |

## Example: Taylor-Green Vortex

A complete 2D periodic Taylor-Green vortex simulation (from `tutorials/seg/ex1.c`):

```c
#include <flucaphys.h>
#include <flucaseg.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <petscmath.h>

static const char help[] = "2D Taylor-Green vortex with Seg SRK\n";

/* Fill solution vector with exact TGV solution at time t */
static PetscErrorCode FillExactSolution(DM dm, PetscReal nu, PetscReal t, Vec Y)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  PetscInt            xs, ys, xm, ym, slot_elem, i, j, c;
  PetscReal           decay_v, decay_p;

  PetscFunctionBegin;
  decay_v = PetscExpReal(-2. * nu * t);
  decay_p = PetscExpReal(-4. * nu * t);

  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_elem));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      PetscReal     x = PetscRealPart(arrc[0][i][slot_elem]);
      PetscReal     y = PetscRealPart(arrc[1][j][slot_elem]);
      PetscScalar   vals[3];
      DMStagStencil stencils[3];

      vals[0] = -PetscCosReal(x) * PetscSinReal(y) * decay_v;
      vals[1] = PetscSinReal(x) * PetscCosReal(y) * decay_v;
      vals[2] = -0.25 * (PetscCosReal(2. * x) + PetscCosReal(2. * y)) * decay_p;

      for (c = 0; c < 3; c++) {
        stencils[c].i   = i;
        stencils[c].j   = j;
        stencils[c].k   = 0;
        stencils[c].loc = DMSTAG_ELEMENT;
        stencils[c].c   = c;
      }
      PetscCall(DMStagVecSetValuesStencil(dm, Y, 3, stencils, vals, INSERT_VALUES));
    }
  }

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sol_dm;
  Phys      phys;
  Seg       seg;
  Vec       Y;
  PetscReal rho = 1., mu = 0.01, nu;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rho", &rho, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &mu, NULL));
  nu = mu / rho;

  /* 1. Create 2D periodic base DMStag (3 element DOFs: u, v, p) */
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC,
                           32, 32,
                           PETSC_DECIDE, PETSC_DECIDE,
                           0, 0, 1,             /* 0 vertex, 0 face, 1 element DOF per direction */
                           DMSTAG_STENCIL_STAR, 1,
                           NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2. * PETSC_PI, 0., 2. * PETSC_PI, 0., 0.));

  /* 2. Create physics model */
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSINS));
  PetscCall(PhysSetBaseDM(phys, dm));
  PetscCall(PhysINSSetDensity(phys, rho));
  PetscCall(PhysINSSetViscosity(phys, mu));
  PetscCall(PhysSetFromOptions(phys));
  PetscCall(PhysSetUp(phys));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));

  /* 3. Create segregated time integrator */
  PetscCall(SegCreate(PETSC_COMM_WORLD, &seg));
  PetscCall(SegSetType(seg, SEGSRK));
  PetscCall(PhysSetUpSeg(phys, seg));  /* wires FD operators from Phys to Seg */

  /* 4. Set initial condition and time parameters */
  PetscCall(DMCreateGlobalVector(sol_dm, &Y));
  PetscCall(FillExactSolution(sol_dm, nu, 0., Y));
  PetscCall(SegSetSolution(seg, Y));
  PetscCall(SegSetTimeStepSize(seg, 0.02));
  PetscCall(SegSetMaxTime(seg, 0.1));
  PetscCall(SegSetFromOptions(seg));
  PetscCall(SegSetUp(seg));

  /* 5. Solve */
  PetscCall(SegSolve(seg));

  /* 6. Destroy in reverse order */
  PetscCall(VecDestroy(&Y));
  PetscCall(SegDestroy(&seg));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
}
```

### Running

```bash
# Default: 32x32 grid, dt=0.02, T=0.1
./tgv

# Finer grid, different scheme
./tgv -stag_grid_x 64 -stag_grid_y 64 -seg_srk_type ark436l2sa

# Longer simulation
./tgv -seg_dt 0.01 -seg_max_time 1.0
```

## Command-Line Options

### Seg (time integration)

| Option | Description | Default |
|--------|-------------|---------|
| `-seg_type <type>` | Time integrator type | `srk` |
| `-seg_dt <real>` | Time step size | (must be set) |
| `-seg_max_time <real>` | Final time | (must be set) |
| `-seg_max_steps <int>` | Maximum number of steps | `PETSC_INT_MAX` |
| `-seg_srk_type <name>` | IMEX RK scheme | `ars343` |

### Grid (from PETSc DMStag)

| Option | Description | Default |
|--------|-------------|---------|
| `-stag_grid_x <int>` | Grid cells in x-direction | (set at creation) |
| `-stag_grid_y <int>` | Grid cells in y-direction | (set at creation) |
| `-stag_grid_z <int>` | Grid cells in z-direction | (set at creation) |

### Solver (from PETSc KSP/PC)

PETSc solver options can be passed through to the internal Helmholtz and Poisson solvers. Refer to [PETSc documentation](https://petsc.org/release/manual/) for details.

## Output and Visualization

### CGNS Output

Fluca outputs solution data in CGNS format:

```c
PetscViewer viewer;
PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, "output.cgns", FILE_MODE_WRITE, &viewer));
/* Write solution vectors via viewer */
PetscCall(PetscViewerDestroy(&viewer));
```

### Loading Solutions

```c
PetscViewer viewer;
PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, "input.cgns", FILE_MODE_READ, &viewer));
PetscCall(FlucaVecLoad(vec, viewer));
PetscCall(PetscViewerDestroy(&viewer));
```

### Visualizing with ParaView

1. Open ParaView
2. File > Open > Select the `.cgns` file
3. Click "Apply" in the Properties panel
4. Select variables to visualize (velocity, pressure)
5. Use filters for streamlines, contours, etc.

## Next Steps

- Browse tutorials in `fluca/tutorials/fd/` for FlucaFD operator examples
- Browse tutorials in `fluca/tutorials/seg/` for time integration examples
- Read the [THEORY_GUIDE.md](THEORY_GUIDE.md) for mathematical background
- Refer to [PETSc documentation](https://petsc.org/release/manual/) for advanced solver options
