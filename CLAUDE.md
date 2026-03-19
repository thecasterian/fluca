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
├── include/           Public headers (flucafd.h, flucamesh.h, flucans.h, ...)
│   └── fluca/private/ Implementation headers (*impl.h)
├── src/
│   ├── sys/           FlucaInitialize/Finalize, shared utilities
│   ├── fd/            Finite difference operators (FlucaFD) on DMStag
│   │   ├── interface/ Base class (create, setup, apply, options)
│   │   └── impls/     Subtypes: derivative, composition, scale, sum, secondordertvd
│   ├── mesh/          Mesh abstraction (Mesh, MeshCart)
│   ├── ns/            Navier-Stokes solver (NS) with segregated methods
│   ├── seg/           Segregated solver framework
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
- **Mesh / MeshCart**: Cartesian grid with boundary types, coordinate setup, and DM access.
- **NS**: Incompressible Navier-Stokes solver — Crank-Nicolson time integration, fractional step pressure-velocity coupling.
- **Viewer**: CGNS file I/O for solution data.
