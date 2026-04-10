# Fluca Codemap

## Project Structure

```
fluca/
├── CMakeLists.txt                  Root build file
├── README.md                       Project overview
├── docs/
│   ├── QUICK_START.md              Getting started guide
│   ├── THEORY_GUIDE.md             Mathematical formulation
├── cmake/                          CMake modules
│   ├── FlucaTestUtils.cmake        Test registration (fluca_parse_test_file, fluca_parse_tutorial_file)
│   ├── RunTest.cmake               Golden-output test runner
│   └── RunTutorial.cmake           Tutorial test runner (exit code only)
├── docs/                           Design documents
│   └── CODEMAPS/                   This codemap
└── fluca/                          Main source tree
    ├── include/                    Public headers
    ├── src/                        Library sources
    ├── tests/                      Unit tests (golden-output)
    └── tutorials/                  Example programs
```

## Library Modules

```
fluca::sys ─────────────────────────────────────────────────────────
  FlucaInitialize / FlucaFinalize, package registration

fluca::fd ──────────── depends on: fluca::sys ──────────────────────
  FlucaFD polymorphic finite difference operators on DMStag

fluca::mesh ────────── depends on: fluca::sys ──────────────────────
  Mesh / MeshCart Cartesian grid abstraction

fluca::seg ─────────── depends on: fluca::fd ───────────────────────
  Seg segregated time integrator, SegSRK subtype

fluca::phys ────────── depends on: fluca::fd, fluca::seg ──────────
  Phys physics model, PhysINS incompressible Navier-Stokes

fluca::viewer ──────── depends on: fluca::sys ──────────────────────
  CGNS file I/O via PetscViewer
```

### Dependency Graph

```
                    fluca::sys
                   /    |     \
            fluca::fd  fluca::mesh  fluca::viewer
               |
            fluca::seg
               |
            fluca::phys (also depends on fluca::fd)
```

## Module Details

### sys — System Utilities

```
include/
  flucasys.h                    FlucaInitialize, FlucaFinalize, FLUCA_EXTERN macro

src/sys/
  flucainit.c                   Initialize/Finalize implementation
  flucapkg.c                    Package registration (registers all modules)
```

### fd — Finite Difference Operators

```
include/
  flucafd.h                     FlucaFD public API, all subtypes, enums, stencil types
  fluca/private/
    fdimpl.h                    FlucaFD ops table and base struct
    fdderivativeimpl.h          FlucaFD_Derivative data
    fdcompositionimpl.h         FlucaFD_Composition data
    fdscaleimpl.h               FlucaFD_Scale data
    fdsumimpl.h                 FlucaFD_Sum data
    fdsecondordertvdimpl.h      FlucaFD_SecondOrderTVD data

src/fd/
  interface/
    fdbasic.c                   Create, SetType, SetUp, Destroy, View
    fdopts.c                    SetFromOptions, SetDM, Set/GetInputLocation, Set/GetOutputLocation
    fdapply.c                   Apply, ApplyDot, GetOperator, GetStencil, GetStencilRaw
    fdpkg.c                     FlucaFDInitializePackage / FinalizePackage
    fdreg.c                     FlucaFDRegister, FlucaFDRegisterAll
  impls/
    derivative/
      derivative.c              1st/2nd order FD in one direction; stencil computation with BCs
    composition/
      composition.c             A ∘ B operator; stencil merging
    scale/
      scale.c                   Constant or vector scaling of operator output
    sum/
      sum.c                     Summation of multiple operators
    secondordertvd/
      secondordertvd.c          TVD convection with mass flux and limiter
      secondordertvdlimiter.c   Limiter functions: minmod, vanleer, superbee, mc, vanalbada
  utils/
    fdutils.c                   Shared utilities (coordinate helpers, etc.)
```

**Key design**: FlucaFD is a PETSc-style polymorphic class. Each subtype implements `GetStencilRaw` to return stencil weights at a grid point. The base class `Apply` iterates over the grid, calls the stencil function, and accumulates into the output vector. `GetOperator` assembles a sparse matrix from the stencils.

### mesh — Mesh Abstraction

```
include/
  flucamesh.h                   Mesh base API (Create, SetUp, GetDM, etc.)
  flucameshcart.h               MeshCart API (Create2d/3d, coordinates, boundaries)
  fluca/private/
    meshimpl.h                  Mesh ops table and base struct
    meshcartimpl.h              MeshCart data (DMs, coordinates, boundary info)

src/mesh/
  interface/
    meshbasic.c                 Create, SetType, SetUp, Destroy, View, GetDM
    meshopts.c                  SetFromOptions
    meshpkg.c                   MeshInitializePackage / FinalizePackage
    meshreg.c                   MeshRegister, MeshRegisterAll
  impl/cart/
    cart.c                      MeshCart implementation (Create2d/3d, coordinates, boundaries)
    cartcgns.c                  CGNS mesh I/O
    cartvec.c                   Vector creation helpers
```

**Key design**: MeshCart creates and manages multiple PETSc DMStag objects (scalar, vector, stag_scalar, stag_vector) for different field layouts. Provides coordinate access and boundary index queries.

### phys — Physics Models

```
include/
  flucaphys.h                   Phys base + PhysINS API, BC types, body force callback
  fluca/private/
    physimpl.h                  Phys ops table (setup, setupseg, etc.)
    physinsimpl.h               Phys_INS data (FD operators, BCs, IS arrays)

src/phys/
  interface/
    physbasic.c                 Create, SetType, SetUp, Destroy, View, PhysSetUpSeg
    physopts.c                  SetFromOptions
    physpkg.c                   PhysInitializePackage / FinalizePackage
    physreg.c                   PhysRegister, PhysRegisterAll
  impls/ins/
    ins.c                       PhysINS Create/Setup/Destroy; constructs all FD operators
    insops.c                    PhysSetUpSeg_INS (wires operators to Seg); SRKExplicitRHS callback
```

**Key design**: PhysINS constructs FlucaFD operators for Laplacian, gradient, divergence, stabilization, and convection during `PhysSetUp`. `PhysSetUpSeg` then passes these operators to the Seg time integrator. The `SRKExplicitRHS` callback evaluates convection + source (no pressure gradient) at each SRK stage.

### seg — Segregated Time Integrator

```
include/
  flucaseg.h                    Seg base + SegSRK API, RHS callback, 12 scheme names
  fluca/private/
    segimpl.h                   Seg ops table and base struct
    segsrkimpl.h                Seg_SRK data (tableau, stage vectors, mu_tilde, KSPs)

src/seg/
  interface/
    segbasic.c                  Create, SetType, SetUp, Reset, Destroy, View
    segopts.c                   SetFromOptions, time/step setters
    segsol.c                    SegStep, SegSolve (time loop)
    segreg.c                    SegRegister, SegRegisterAll
    segpkg.c                    SegInitializePackage (calls SegSRKInitializePackage)
  impls/srk/
    srk.c                       SegSRK Create/Setup/Destroy; operator setters; matrix assembly
    srkstep.c                   SegStep_SRK: stage loop, Helmholtz/Poisson solves, mu/mu_tilde
    srktab.h                    SRKTableau struct and linked-list node (private)
    srktab.c                    Tableau registry: 12 schemes, register/lookup/init/finalize
```

**Key design**: Seg is physics-agnostic — it depends on `fluca::fd`, not `fluca::phys`. It receives FlucaFD operator handles via setter functions (called by `PhysSetUpSeg`). SegSRK manages its own KSPs and matrices, assembled from the FlucaFD operators. The tableau registry follows PETSc's ARKIMEX pattern with a global linked list.

### viewer — CGNS I/O

```
include/
  flucaviewer.h                 PetscViewerFlucaCGNS API, FlucaVecLoad

src/viewer/
  interface/
    viewerbasic.c               FlucaOptionsCreateViewer, FlucaObjectViewFromOptions
    viewervec.c                 FlucaVecLoad implementation
  impl/flucacgns/
    flucacgns.c                 CGNS viewer: Open, read/write fields, batch I/O
```

## Tests

```
tests/
  CMakeLists.txt                Includes fd/, phys/, seg/ subdirectories
  fd/
    CMakeLists.txt              FD operator tests
    ex1.c ... ex7.c             Derivative, composition, scale, sum, TVD tests
    fdtest.h                    Shared test helpers
    output/                     Golden output files
  phys/
    CMakeLists.txt              Physics model tests
    ex1.c ... ex4.c             INS operator, residual, Jacobian tests
    output/
  seg/
    CMakeLists.txt              Seg lifecycle tests
    ex1.c                       SRK lifecycle (12 test cases, one per scheme)
    ex2.c, ex3.c                Additional infrastructure tests
    output/
```

Tests use golden-output comparison: each test prints deterministic values to stdout, compared byte-exact against `.out` files. Test cases are defined in `/*TEST*/` blocks parsed by CMake at configure time.

## Tutorials

```
tutorials/
  CMakeLists.txt                Includes fd/, seg/ subdirectories
  fd/
    CMakeLists.txt              FD tutorial builds
    ex1.c                       1D steady convection-diffusion with TVD
    ex2.c                       2D Poisson equation
    ex3.c                       1D unsteady advection with TVD
    ex4.c                       2D unsteady advection with TVD
  seg/
    CMakeLists.txt              Seg tutorial builds
    ex1.c                       2D Taylor-Green vortex with SRK
```

Tutorial tests check exit code 0 only (no golden output comparison). They are slow and should only be run when verifying tutorial-related changes.

## Typical Data Flow

```
User code
  │
  ├─ DMStagCreate2d(...)              Create PETSc DMStag grid
  │
  ├─ PhysCreate → PhysSetType(PHYSINS)
  │   ├─ PhysSetBaseDM(dm)
  │   ├─ PhysINSSetDensity / SetViscosity
  │   ├─ PhysINSSetBoundaryCondition   (for non-periodic)
  │   ├─ PhysSetBodyForce              (optional)
  │   └─ PhysSetUp                     Constructs all FlucaFD operators internally
  │
  ├─ SegCreate → SegSetType(SEGSRK)
  │   ├─ PhysSetUpSeg(phys, seg)       Wires FD operators + RHS callback from Phys to Seg
  │   ├─ SegSetSolution(Y)
  │   ├─ SegSetTimeStepSize / SetMaxTime
  │   └─ SegSetUp                      Assembles Helmholtz + Poisson matrices
  │
  ├─ SegSolve(seg)                     Time loop: stages → Helmholtz → Poisson → update
  │
  └─ Destroy in reverse order
```
