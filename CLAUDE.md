# Fluca

CFD framework for incompressible viscous flows on Cartesian grids, built on PETSc. Uses collocated grid with Rhie-Chow interpolation, fractional step method, and finite difference spatial discretization.

## Language and Build

Pure C project (not C++). Follows PETSc coding conventions — see `petsc-conventions` skill for details.

- **Memory**: `PetscNew`/`PetscMalloc1`/`PetscFree` — no `malloc`/`free`, no C++ smart pointers
- **Error handling**: `PetscCall()`, `PetscCheck()` — no exceptions, no `assert()`
- **Testing**: Golden-output comparison via `ctest` — not googletest
- **Build**: `cmake --build build && ctest --test-dir build`
- **Dependencies**: PETSc >= 3.23, HDF5, CGNS (parallel I/O)

## Source Layout

```
fluca/
├── include/           Public headers (flucafd.h, flucaphys.h, flucaseg.h, ...)
│   └── fluca/private/ Implementation headers (*impl.h)
├── src/
│   ├── sys/           FlucaInitialize/Finalize, shared utilities
│   ├── fd/            Finite difference operators (FlucaFD) on DMStag
│   │   ├── interface/ Base class (create, setup, apply, options)
│   │   └── impls/     Subtypes: derivative, composition, scale, sum, secondordertvd
│   ├── phys/          Physical problem statement (Phys) — fields, BCs, properties
│   │   ├── interface/ Base class, field registry, property registry
│   │   └── impls/     Subtypes: laminar
│   ├── seg/           Segregated solver (Seg) — discretization and solution
│   │   ├── interface/ Base class, time loop, monitors
│   │   ├── utils/     Operator construction (ops), PCABF preconditioner (abfpc)
│   │   └── impls/     Subtypes: cnlinear
│   └── viewer/        CGNS I/O via PetscViewer
├── tests/             Golden-output tests (ex*.c + output/*.out)
├── tutorials/         Example programs
└── cmake/             FlucaTestUtils, RunTest helpers
```

## Git Workflow

- Trunk-based development
- Never commit directly to `main` — always create a new branch from the latest `main` and open a PR
- Branch names must use a prefix: `feature/`, `refactor/`, `test/`, `doc/`, etc
- When committing, only include source code files (`.c`, `.h`), build files (`CMakeLists.txt`), and test output files (`.out`). Never include documentation, `.claude/` files, or other non-source files unless explicitly requested.

## Agent Requirements

- Any agent that modifies code in this project **must** verify its changes against the `petsc-conventions` skill. When spawning such agents, tell them to read `.claude/skills/petsc-conventions/SKILL.md` so they can self-check before returning results.
- Any agent that adds tests **must** follow the `add-test` skill workflow. When spawning such agents, tell them to read `.claude/skills/add-test/SKILL.md` so they handle source generation, CMakeLists.txt registration, golden output capture, and ctest verification correctly.

## Key Modules

- **FlucaFD**: Polymorphic finite difference operator on PETSc DMStag. Subtypes compute stencils for derivatives, compositions, scaling, sums, and TVD schemes.
- **Phys**: States the continuous problem and nothing else — which fields exist (with an equation role each), the solution DM, per-field boundary conditions, material properties, and null-space declarations. It owns no stencils, no matrices, and no time step; its ops table has no `compute*` entry. Subtype `PHYSLAMINAR` is isothermal laminar incompressible flow.
- **Seg**: Discretizes and solves the problem a `Phys` states. Owns operator construction, the time discretization, the coupled linear solve and the time loop. Subtype `SEGCNLINEAR` is the linearized Crank-Nicolson scheme with incremental pressure, preconditioned by `PCABF` (the fractional step method expressed as an approximate block factorization).
- **Viewer**: CGNS file I/O for solution data.

There is no application target; `Seg` is driven from tutorials and tests.
