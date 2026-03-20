# NS Redesign Plan

## Context

FlucaFD (finite difference operator on DMStag) is fully implemented. The current NS module uses hardcoded stencils (~5000 lines), a custom ABF preconditioner, manual Crank-Nicolson timestepping, 4 separate DMStag objects via Mesh, and DMComposite/VecNest/MatNest. This redesign replaces all of that.

### Architecture change

```
Before:  User -> NS -> (TS/SNES internally, Mesh, DMComposite, VecNest, MatNest, ABF)
After:   User -> TS directly, Phys provides operators + callbacks
```

The NS module is removed entirely. A new **Phys** (physical model) class replaces it. Phys is a polymorphic PETSc-style class whose subtypes represent different governing equations. Each subtype constructs FlucaFD operators and provides callback functions (`IFunction`/`IJacobian`/`RHSFunction` for TS). The user creates and drives TS directly.

### Decisions

- **DM separation**: User provides a **base DM** (grid topology + coordinates) via `PhysSetBaseDM()`. Phys creates its own **solution DM** with the correct DOFs for its subtype during `PhysSetUp()`. Retrieved via `PhysGetSolutionDM()`. This decouples grid specification (user's concern) from DOF layout (physical model's concern).
- **Formulation**: Monolithic IMEX (all unknowns solved simultaneously at each implicit stage). The incompressible NS equations are transformed from a DAE into a first-order ODE by applying the `alpha + d/dt` operator to the pressure-stabilized continuity equation, following [Bakhvalov 2025](https://arxiv.org/abs/2506.09519). All operators are affine (boundary contributions baked in); no separate H(t) term. Convection is treated explicitly (RHSFunction) and diffusion + pressure + continuity implicitly (IFunction).
- **Time integration**: TSARKIMEX (additive Runge-Kutta IMEX). Stiffly accurate schemes recommended. Default: `TSARKIMEX3` (3rd order, L-stable). User controls via `-ts_type`, `-ts_arkimex_type`.
- **Solver**: User creates TS directly. Phys wires callbacks via `PhysSetUpTS()`. No separate SNES path — steady-state problems use `TSPSEUDO` or solve as the limit of an unsteady problem.
- **Phys subtypes**: Polymorphic by physical model. For now, only `PHYSINS` (incompressible Navier-Stokes). Different subtypes define different DOF layouts on the solution DM (e.g., INS: dim+1 element DOFs; Boussinesq: dim+2).
- **Spatial discretization**: FlucaFD operators replace all hardcoded stencils. FlucaFD already supports function-based BC values and per-component boundary conditions on `main`. A new `FlucaFDApplyDot` function and `fn_dot` field in `FlucaFDBoundaryCondition` are added as a prerequisite (see Phase 0) to support the ODE-transformed continuity equation, where operators must be applied to time derivatives with distinct ghost fill.
- **Preconditioner**: Default is ILU; user can configure `PCFIELDSPLIT` or any other PC via the options database.
- **Parameters**: Subtype-specific setters (`PhysINSSetDensity`, `PhysINSSetViscosity`) plus options database (`-phys_ins_density`, `-phys_ins_viscosity`).
- **Mesh module**: Left untouched; removed in a separate follow-up task.
- **Convection**: Second-order TVD scheme with configurable flux limiter for upwind-biased convection. Mass flux (rho * u) used at faces, not bare velocity.

### Reference implementation

The `feature/ns-redesign` branch contains a working implementation of Phases 0--3 from the original plan. This revised plan is written for implementation on `main`. Where the branch's design differs from this plan, this plan has priority.

---

## Formulation

### Governing equations (semidiscrete)

The incompressible Navier-Stokes equations are discretized in space to give:

```
Momentum:    rho * du/dt + F_conv(u) + Gp = mu * L * u + f(t)
Continuity:  Du + sigma_0 * S * p = 0
```

where:

| Symbol | Meaning | Fluca operator |
|--------|---------|----------------|
| `u` | cell-centered velocity (dim components) | solution DOFs 0..dim-1 |
| `p` | cell-centered pressure (1 component) | solution DOF dim |
| `G` | pressure gradient | `fd_grad_p[d]` |
| `L` | viscous Laplacian (L < 0) | via `fd_laplacian[d]` |
| `D` | divergence of interpolated velocity | `fd_div` (includes rho factor) |
| `S` | pressure stabilization = DTG - DG^st >= 0 | `fd_pstab` (includes sigma_0 factor) |
| `F_conv` | nonlinear convection | `fd_conv[d]` |
| `f(t)` | body force | user callback |

All spatial operators (D, S, G, L) are **affine** — boundary contributions are baked in via FlucaFD boundary conditions. There is no separate H(t) term; the operators handle boundary values internally during ghost fill.

The pressure stabilization S arises from collocated (Rhie-Chow) discretization. It is the difference between the "wide" Laplacian DTG (cell -> face interpolation -> face gradient -> divergence) and the "compact" staggered Laplacian DG^st (direct face gradient -> divergence). The stabilization parameter is `sigma_0 = dt / rho`, matching the classical Rhie-Chow coefficient.

### ODE transformation

With `sigma_0 > 0` and `sigma_1 = 0`, the continuity equation is algebraic (no dp/dt), making the system a DAE. Following [Bakhvalov 2025, Sec. 5.3], we apply the operator `alpha + d/dt` (where `alpha = 1/dt`) to the continuity equation to obtain a first-order ODE:

```
Momentum:    rho * du/dt + F_conv(u) + Gp = mu * L * u + f(t)
Continuity': alpha * D(u) + D_dot(du/dt) + alpha * sigma_0 * S(p) + sigma_0 * S_dot(dp/dt) = 0
```

where `D(·)` and `S(·)` denote affine operators applied with value BCs (using `FlucaFDApply`), while `D_dot(·)` and `S_dot(·)` denote the same operators applied with time-derivative BCs (using `FlucaFDApplyDot`). The `_dot` variants compute time derivatives of BC values automatically: zero for constant BCs, central FD approximation for time-dependent `fn`, or exact `fn_dot` if provided.

The `alpha * D(u) + alpha * sigma_0 * S(p)` terms act as constraint feedback: they drive any violation of the undifferentiated continuity equation to zero with rate `alpha`. The `D_dot(du/dt) + sigma_0 * S_dot(dp/dt)` terms enforce the time derivative of the constraint.

Both du/dt and dp/dt now appear, giving a proper ODE with mass matrix:

```
M * dy/dt = rhs,   y = (u, p)^T

    [ rho * I        0        ]
M = [                          ]
    [ rho * D    sigma_0 * S  ]
```

M is lower-triangular and invertible (given sigma_0 > 0 and proper BCs), so the system is a well-posed ODE.

Key properties:
- The `alpha + d/dt` operator combines constraint feedback and time differentiation into a single expression — no separate Xi computation or dH/dt term is needed.
- Without the `alpha` terms, only d/dt(constraint) = 0 is enforced — initial constraint errors persist. The feedback makes errors decay as ~exp(-alpha * t).
- Because all operators are affine (boundary contributions baked in), boundary terms are handled automatically through the ghost fill mechanism. `FlucaFDApply` uses value BCs; `FlucaFDApplyDot` uses time-derivative BCs.

### IMEX splitting for TSARKIMEX

The ODE is split into implicit (stiff) and explicit (non-stiff) parts. The entire continuity row is implicit — the `alpha + d/dt` formulation places all continuity terms in the IFunction:

**IFunction** `F(t, y, y_dot)`:
```
F_momentum   = rho * u_dot - mu * L * u + Gp
F_continuity = alpha * D(u) + D_dot(u_dot) + alpha * sigma_0 * S(p) + sigma_0 * S_dot(p_dot)
```

where `D(·)`, `S(·)` use `FlucaFDApply` (value BCs) and `D_dot(·)`, `S_dot(·)` use `FlucaFDApplyDot` (time-derivative BCs).

**RHSFunction** `G(t, y)`:
```
G_momentum   = -F_conv(u) + f(t)
G_continuity = 0
```

**IJacobian** `dF/dy + shift * dF/dy_dot`:
```
[       shift * rho * I - mu * L                G               ]
[                                                                 ]
[ (shift + alpha) * rho * D    (shift + alpha) * sigma_0 * S     ]
```

The continuity row uses `shift + alpha` because the IFunction depends on both `y` (via the `alpha * D(u)` and `alpha * sigma_0 * S(p)` feedback terms) and `y_dot` (via the `D_dot(u_dot)` and `S_dot(p_dot)` terms). The Jacobian matrices from `FlucaFDGetOperator` are the same for both — BC values only affect the affine (constant) term, which drops out of the Jacobian.

**RHSJacobian** `dG/dy`:
```
[ d(-F_conv)/du    0 ]
[                    ]
[       0          0 ]
```

The Picard-linearized convection Jacobian (frozen velocity) is used for `d(-F_conv)/du`.

Note: for constant viscosity, the IJacobian is constant for fixed `shift` and `alpha`. Each implicit TSARKIMEX stage solves a linear system — SNES converges in one iteration with `-snes_type ksponly`.

### Stabilization parameter update

`sigma_0 = dt / rho` depends on the time step. The `fd_pstab` operator's scale factor and `alpha` must be updated whenever dt changes. This is done in a `TSPreStep` callback:

1. Query `TSGetTimeStep(ts, &dt)`
2. Update `fd_pstab` scale factor to `dt / rho`
3. Update `alpha = 1 / dt`

Fixed dt (`-ts_adapt_type none`) is recommended initially since varying sigma_0 introduces O(dt) perturbations to the stabilization.

---

## User flow example

```c
/* Create base DM (grid topology + coordinates only, DOFs don't matter) */
DMStagCreate2d(comm, bx, by, M, N, ..., 0, 0, 1, stencil, width, NULL, NULL, &base_dm);
DMStagSetUniformCoordinatesProduct(base_dm, ...);

/* Create physical model */
PhysCreate(comm, &phys);
PhysSetType(phys, PHYSINS);
PhysSetBaseDM(phys, base_dm);
PhysINSSetDensity(phys, rho);
PhysINSSetViscosity(phys, mu);
PhysINSSetBoundaryCondition(phys, 0 /* left */, bc);
/* ... more BCs ... */
PhysSetFromOptions(phys);
PhysSetUp(phys);   /* creates solution DM with correct DOFs internally */

/* Get the solution DM (has dim+1 element DOFs for INS) */
DM sol_dm;
PhysGetSolutionDM(phys, &sol_dm);

/* Create and configure TS */
TSCreate(comm, &ts);
PhysSetUpTS(phys, ts);   /* sets DM, wires IFunction/IJacobian/RHSFunction */
TSSetFromOptions(ts);     /* user controls -ts_type, -ts_dt, -ts_max_time */

/* Solve */
DMCreateGlobalVector(sol_dm, &sol);
/* ... set initial condition ... */
TSSolve(ts, sol);

/* Cleanup */
VecDestroy(&sol);
TSDestroy(&ts);
PhysDestroy(&phys);
DMDestroy(&base_dm);
```

---

## Public API (`flucaphys.h`)

All Phys API — including INS-specific types and functions — lives in a single header.

```c
typedef struct _p_Phys *Phys;
typedef const char *PhysType;

#define PHYSINS "ins"   /* Incompressible Navier-Stokes */

FLUCA_EXTERN PetscClassId PHYS_CLASSID;

/* Body force callback */
typedef PetscErrorCode PhysBodyForceFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx);

/* INS boundary condition types */
typedef enum {
  PHYS_INS_BC_NONE,
  PHYS_INS_BC_VELOCITY,
} PhysINSBCType;

/* INS boundary condition callback: returns value of field component at boundary coordinates.
   comp is the solution DOF component being queried (0..dim-1 for velocity, dim for pressure). */
typedef PetscErrorCode PhysINSBCFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx);

typedef struct {
  PhysINSBCType  type;
  PhysINSBCFn   *fn;       /* value BC: u_bc(t, x, comp) */
  void          *ctx;
  PhysINSBCFn   *fn_dot;   /* time derivative BC: du_bc/dt(t, x, comp); NULL = use FD approx of fn */
  void          *fn_dot_ctx;
} PhysINSBC;

/* Lifecycle */
FLUCA_EXTERN PetscErrorCode PhysCreate(MPI_Comm, Phys *);
FLUCA_EXTERN PetscErrorCode PhysSetType(Phys, PhysType);
FLUCA_EXTERN PetscErrorCode PhysGetType(Phys, PhysType *);
FLUCA_EXTERN PetscErrorCode PhysSetBaseDM(Phys, DM);
FLUCA_EXTERN PetscErrorCode PhysGetBaseDM(Phys, DM *);
FLUCA_EXTERN PetscErrorCode PhysGetSolutionDM(Phys, DM *);
FLUCA_EXTERN PetscErrorCode PhysSetFromOptions(Phys);
FLUCA_EXTERN PetscErrorCode PhysSetUp(Phys);
FLUCA_EXTERN PetscErrorCode PhysDestroy(Phys *);
FLUCA_EXTERN PetscErrorCode PhysView(Phys, PetscViewer);
FLUCA_EXTERN PetscErrorCode PhysViewFromOptions(Phys, PetscObject, const char[]);

/* Options prefix */
FLUCA_EXTERN PetscErrorCode PhysSetOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysAppendOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysGetOptionsPrefix(Phys, const char *[]);

/* Body force (base class) */
FLUCA_EXTERN PetscErrorCode PhysSetBodyForce(Phys, PhysBodyForceFn *, void *);

/* Solver setup (TS only) */
FLUCA_EXTERN PetscErrorCode PhysSetUpTS(Phys, TS);

/* Direct residual/Jacobian access (for testing) */
FLUCA_EXTERN PetscErrorCode PhysComputeIFunction(Phys, PetscReal, Vec, Vec, Vec);
FLUCA_EXTERN PetscErrorCode PhysComputeIJacobian(Phys, PetscReal, Vec, Vec, PetscReal, Mat, Mat);
FLUCA_EXTERN PetscErrorCode PhysComputeRHSFunction(Phys, PetscReal, Vec, Vec);

/* PHYSINS specific */
FLUCA_EXTERN PetscErrorCode PhysINSSetDensity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysINSGetDensity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysINSSetViscosity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysINSGetViscosity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysINSSetBoundaryCondition(Phys, PetscInt, PhysINSBC);
FLUCA_EXTERN PetscErrorCode PhysINSGetBoundaryCondition(Phys, PetscInt, PhysINSBC *);

/* Registration */
FLUCA_EXTERN PetscFunctionList PhysList;
FLUCA_EXTERN PetscErrorCode    PhysRegister(const char[], PetscErrorCode (*)(Phys));
```

### Design notes

**No separate `flucaphysins.h`**: INS-specific types and functions are in `flucaphys.h`. This keeps the user-facing include simple.

**No `PhysBoundaryFace` enum**: Boundary faces are identified by `PetscInt` index (0=left, 1=right, 2=down, 3=up, 4=back, 5=front), matching the FlucaFD convention.

**No SNES path**: `PhysSetUpSNES` was removed during the reference implementation. TS subsumes SNES — steady-state problems can use `TSPSEUDO` or simply run TS to steady state.

**BC callback adapter**: The `PhysINSBCFn` signature `(dim, t, x[], comp, *val, ctx)` includes `comp`, but `FlucaFDBCValueFn` signature `(dim, t, x[], ctx, *value)` does not. The INS subtype bridges this with a small adapter struct per boundary face that captures the component and calls the user's `PhysINSBCFn`. Since `main` already supports per-component BCs via `FlucaFDSetBoundaryConditions(fd, comp, bcs[])`, the adapter is set per-component on each FlucaFD operator.

---

## Internal data structures

### Base class (`physimpl.h`)

```c
struct _PhysOps {
  PetscErrorCode (*setfromoptions)(Phys, PetscOptionItems);
  PetscErrorCode (*setup)(Phys);
  PetscErrorCode (*destroy)(Phys);
  PetscErrorCode (*view)(Phys, PetscViewer);
  PetscErrorCode (*createsolutiondm)(Phys);
  PetscErrorCode (*setupts)(Phys, TS);
  PetscErrorCode (*computeifunction)(Phys, PetscReal, Vec, Vec, Vec);
  PetscErrorCode (*computeijacobian)(Phys, PetscReal, Vec, Vec, PetscReal, Mat, Mat);
  PetscErrorCode (*computerhsfunction)(Phys, PetscReal, Vec, Vec);
  PetscErrorCode (*computerhsjacobian)(Phys, PetscReal, Vec, Mat, Mat);
};

struct _p_Phys {
  PETSCHEADER(struct _PhysOps);

  /* Parameters */
  DM               base_dm;      /* user-provided DMStag (grid topology + coordinates) */
  PhysBodyForceFn *bodyforce;
  void            *bodyforce_ctx;

  /* Data */
  DM       sol_dm;   /* solution DMStag (created by subtype during setup) */
  PetscInt dim;      /* spatial dimension (extracted from base_dm) */
  void    *data;     /* subtype-specific */

  /* State */
  PetscBool setupcalled;
};
```

### INS subtype (`physinsimpl.h`)

```c
#define PHYS_INS_MAX_DIM   3
#define PHYS_INS_MAX_FACES (2 * PHYS_INS_MAX_DIM)

/* Adapter to bridge PhysINSBCFn (has comp) to FlucaFDBCValueFn (no comp) */
typedef struct {
  PhysINSBCFn *fn;       /* value callback */
  PhysINSBCFn *fn_dot;   /* time derivative callback (may be NULL) */
  void        *ctx;
  void        *ctx_dot;
  PetscInt     comp;     /* which solution component this adapter is wired for */
} PhysINS_BCAdapter;

typedef struct {
  PetscReal rho;   /* density */
  PetscReal mu;    /* dynamic viscosity */
  PetscReal alpha; /* constraint feedback parameter = 1/dt */

  /* Boundary conditions (one per face: left, right, down, up, back, front) */
  PhysINSBC bcs[PHYS_INS_MAX_FACES];

  /* BC adapters: [comp][face] — created during setup to bridge PhysINSBCFn to FlucaFDBCValueFn */
  PhysINS_BCAdapter bc_adapters[PHYS_INS_MAX_DIM + 1][PHYS_INS_MAX_FACES];

  /* FlucaFD operators (implicit part) */
  FlucaFD fd_laplacian[PHYS_INS_MAX_DIM]; /* -mu * nabla^2 u_d per velocity direction */
  FlucaFD fd_grad_p[PHYS_INS_MAX_DIM];    /* dp/dx_d per velocity direction */
  FlucaFD fd_div;                         /* rho * div(interp(u)) — single Sum over directions */
  FlucaFD fd_pstab;                       /* sigma_0 * S(p) pressure stabilization */

  /* FlucaFD operators (explicit part — convection) */
  /* C_d = sum_e d/dx_e(F_e * u_d_TVD) where F_e = rho * u_e */
  FlucaFD fd_conv[PHYS_INS_MAX_DIM];                        /* summed convection per velocity dir */
  FlucaFD fd_tvd[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM];       /* TVD interp: [d][e] = u_d along e */
  FlucaFD fd_scale_vel[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* mass flux scaling: [d][e] */
  FlucaFD fd_conv_comp[PHYS_INS_MAX_DIM][PHYS_INS_MAX_DIM]; /* composed conv: [d][e] */
  FlucaFD fd_interp[PHYS_INS_MAX_DIM];                      /* cell-to-face interpolation per dir */
  DM      dm_face[PHYS_INS_MAX_DIM];                        /* face DMs for mass flux */
  Vec     mass_flux_face[PHYS_INS_MAX_DIM];                 /* face mass flux vectors (rho * u) */

  /* Solver data */
  Mat          J;         /* IJacobian matrix */
  Mat          J_rhs;     /* RHSJacobian matrix (Picard convection) */
  IS           is_vel;
  IS           is_p;
  MatNullSpace nullspace;
  Vec          temp;      /* work vector for IFunction computation */
  PetscReal    dt_current; /* current dt for sigma_0 update detection */
  PetscBool    has_pressure_outlet;
} Phys_INS;
```

---

## Phase 0: FlucaFD time-dependent and time-derivative BC support

**Goal**: Make FlucaFD boundary conditions time-aware and add support for applying operators with time-derivative BCs, needed for the ODE-transformed continuity equation.

### Already done on `main`

**`FlucaFDBCValueFn`** — `t` parameter added:
```c
typedef PetscErrorCode (*FlucaFDBCValueFn)(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value);
```

**`FlucaFDApply`** and **`FlucaFDGetStencil`** — `t` parameter added:
```c
PetscErrorCode FlucaFDApply(FlucaFD fd, PetscReal t, DM dm_in, DM dm_out, Vec x, Vec y);
PetscErrorCode FlucaFDGetStencil(FlucaFD fd, PetscReal t, PetscInt i, PetscInt j, PetscInt k, PetscInt *npoints, FlucaFDStencilPoint[]);
```

All internal functions (`FlucaFDEvaluateBCValue_Internal`, `FlucaFDResolveTVDRefs_Internal`) and call sites (tests, tutorials) are updated. `FlucaFDGetOperator` passes `t = 0` since BC values only affect the affine term, not the linear part.

### Remaining: `fn_dot` and `FlucaFDApplyDot`

**`FlucaFDBoundaryCondition`** — add `fn_dot` field:
```c
typedef struct {
  FlucaFDBCType      type;
  PetscScalar        value;
  FlucaFDBCValueFn  *fn;         /* u_bc(t, x) — value BC */
  void              *fn_ctx;
  FlucaFDBCValueFn  *fn_dot;     /* du_bc/dt(t, x) — explicit time derivative (optional) */
  void              *fn_dot_ctx;
} FlucaFDBoundaryCondition;
```

**`FlucaFDApplyDot()`** — new public function:
```c
PetscErrorCode FlucaFDApplyDot(FlucaFD fd, PetscReal t, DM dm_in, DM dm_out, Vec x, Vec y);
```

Same stencil and coefficients as `FlucaFDApply`, but uses time-derivative BC values for ghost fill. The BC value for `ApplyDot` is determined by the following priority:

| `fn` | `fn_dot` | Ghost value |
|------|----------|-------------|
| NULL | NULL | 0 (constant BC has zero time derivative) |
| set | NULL | `(fn(t+h, x) - fn(t-h, x)) / (2h)` with `h = 1e-5` |
| set | set | `fn_dot(t, x)` (exact, no approximation) |

When `fn` is set but `fn_dot` is NULL, the time derivative is approximated by central finite difference using the existing `fn`. This is the default for time-dependent BCs — users only need to provide `fn_dot` if higher accuracy is required or `fn` is not smooth.

For composite operators (Sum, Composition, Scale), the "use dot BCs" mode propagates through the dispatch chain.

### Implementation notes

- The Jacobian matrix from `FlucaFDGetOperator` is unchanged — BC values only affect the affine (constant) term, not the linear part. The same matrix is valid for both `Apply` and `ApplyDot`.
- Internally, the `_dot` mode can be implemented as a flag on the base FlucaFD struct that the ghost fill routine checks. Set before recursive apply, cleared after.

### Verification

Existing tests pass unchanged. Add tests that verify:
- Constant BC (`fn = NULL`): `FlucaFDApplyDot` ghost value is zero.
- Steady function BC (`fn` returns time-independent value): `FlucaFDApplyDot` ghost value is approximately zero.
- Time-dependent function BC (`fn` returns `sin(t) * g(x)`): `FlucaFDApplyDot` ghost value matches `cos(t) * g(x)` within FD tolerance.
- Explicit `fn_dot`: `FlucaFDApplyDot` uses `fn_dot` exactly.

---

## Phase 1: Phys base class scaffold

**Goal**: New `fluca_phys` library builds successfully. No working subtypes yet.

### Source layout

```
fluca/
├── include/
│   ├── flucaphys.h              # Public Phys + INS API (single header)
│   └── fluca/private/
│       ├── physimpl.h           # Base class impl
│       └── physinsimpl.h        # INS subtype impl
├── src/
│   └── phys/
│       ├── CMakeLists.txt
│       ├── interface/
│       │   ├── physbasic.c      # Create, SetUp, Destroy, SetUpTS, ComputeI/RHSFunction/Jacobian
│       │   ├── physopts.c       # SetBaseDM, GetBaseDM, GetSolutionDM, SetFromOptions, prefix
│       │   ├── physpkg.c        # Package init, event registration
│       │   └── physreg.c        # Type registration
│       └── impls/
│           └── ins/             # (Phase 2)
```

### PhysSetUp base logic

1. Validate base DM is DMStag
2. Extract dim from base DM
3. Call subtype `createsolutiondm` op (creates `sol_dm` via `DMStagCreateCompatibleDMStag` with correct DOFs)
4. Call subtype `setup` op (builds FlucaFD operators, determines null space, validates BCs)

### PhysSetUpTS

Dispatch to subtype op:
```c
PetscErrorCode PhysSetUpTS(Phys phys, TS ts)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscCheck(phys->setupcalled, ...);
  PetscUseTypeMethod(phys, setupts, ts);
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### CMakeLists.txt

```cmake
add_library(fluca_phys SHARED
    interface/physbasic.c
    interface/physopts.c
    interface/physpkg.c
    interface/physreg.c
)
target_link_libraries(fluca_phys PUBLIC fluca::fd)
add_library(fluca::phys ALIAS fluca_phys)
```

Add `add_subdirectory(phys)` to `src/CMakeLists.txt`.

### Verification

`cmake --build build` succeeds.

---

## Phase 2: PHYSINS -- full incompressible Navier-Stokes

**Goal**: Complete INS subtype with IMEX time integration via TSARKIMEX. Viscous + pressure terms implicit, convection explicit.

### Files to create

| File | Contents |
|------|----------|
| `include/fluca/private/physinsimpl.h` | Phys_INS struct |
| `src/phys/impls/ins/ins.c` | PhysCreate_INS, Setup, solution DM creation, parameter setters |
| `src/phys/impls/ins/insops.c` | Operator construction, IFunction, IJacobian, RHSFunction, RHSJacobian, TS callbacks |

### Solution DM creation (inside PhysSetUp_INS)

```c
/* INS: dim velocity + 1 pressure at ELEMENT */
DMStagCreateCompatibleDMStag(base_dm, 0, 0, 0, dim + 1, &phys->sol_dm);
```

### BC adapter

The INS subtype wraps `PhysINSBCFn` to `FlucaFDBCValueFn` using `PhysINS_BCAdapter`. The same adapter struct bridges both `fn` (value) and `fn_dot` (time derivative):

```c
static PetscErrorCode PhysINS_BCAdapterFn(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PhysINS_BCAdapter *a = (PhysINS_BCAdapter *)ctx;

  PetscFunctionBeginUser;
  PetscCall(a->fn(dim, t, x, a->comp, value, a->ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysINS_BCAdapterFnDot(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar *value)
{
  PhysINS_BCAdapter *a = (PhysINS_BCAdapter *)ctx;

  PetscFunctionBeginUser;
  PetscCall(a->fn_dot(dim, t, x, a->comp, value, a->ctx_dot));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

When setting BCs on a FlucaFD operator, the INS code wires both `fn` and `fn_dot`:
```c
static PetscErrorCode SetVelocityDirichletBCs(Phys phys, FlucaFD fd, PetscInt comp)
{
  Phys_INS                *ins = (Phys_INS *)phys->data;
  FlucaFDBoundaryCondition fd_bcs[2 * PHYS_INS_MAX_DIM] = {{0}};

  PetscFunctionBegin;
  for (PetscInt f = 0; f < 2 * phys->dim; f++) {
    if (ins->bcs[f].type == PHYS_INS_BC_VELOCITY && ins->bcs[f].fn) {
      ins->bc_adapters[comp][f].fn      = ins->bcs[f].fn;
      ins->bc_adapters[comp][f].fn_dot  = ins->bcs[f].fn_dot;
      ins->bc_adapters[comp][f].ctx     = ins->bcs[f].ctx;
      ins->bc_adapters[comp][f].ctx_dot = ins->bcs[f].fn_dot_ctx;
      ins->bc_adapters[comp][f].comp    = comp;
      fd_bcs[f].type       = FLUCAFD_BC_DIRICHLET;
      fd_bcs[f].fn         = PhysINS_BCAdapterFn;
      fd_bcs[f].fn_ctx     = &ins->bc_adapters[comp][f];
      fd_bcs[f].fn_dot     = ins->bcs[f].fn_dot ? PhysINS_BCAdapterFnDot : NULL;
      fd_bcs[f].fn_dot_ctx = &ins->bc_adapters[comp][f];
    }
  }
  PetscCall(FlucaFDSetBoundaryConditions(fd, comp, fd_bcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Operator construction (inside PhysSetUp_INS)

All operators use `sol_dm` (the solution DM with dim+1 element DOFs). Face locations indexed by direction: `face_loc[] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK}`.

**Viscous Laplacian** (`fd_laplacian[d]`):
```
-mu * nabla^2 u_d = sum_e Composition(Scale(-mu, Derivative(ELEMENT,d -> face[e],0)), Derivative(face[e],0 -> ELEMENT,d))
```
Velocity Dirichlet BCs on the Sum operator (component d).

**Pressure gradient** (`fd_grad_p[d]`):
```
dp/dx_d = Derivative(ELEMENT,dim -> ELEMENT,d)
```
Pressure Neumann (zero) BCs (component dim).

**Divergence** (`fd_div`):
```
rho * div(interp(u)) = rho * sum_d Composition(Derivative(ELEMENT,d -> face[d],0), Derivative(face[d],0 -> ELEMENT,dim))
```
Single Sum operator. Velocity Dirichlet BCs per input component d. The rho factor is baked in so that `fd_div` applied to `y_dot` gives `rho * D * u_dot` for the differentiated continuity.

**Pressure stabilization** (`fd_pstab`):
```
sigma_0 * sum_d (DTG_d - DG^st_d)(p)
  where DG^st_d = Composition(Derivative(ELEMENT,dim -> face[d],0), Derivative(face[d],0 -> ELEMENT,dim))
  and   DTG_d   = Composition(Derivative(ELEMENT,dim -> ELEMENT,d), Composition(Derivative(ELEMENT,d -> face[d],0), Derivative(face[d],0 -> ELEMENT,dim)))
```
Scaled by `sigma_0 = dt / rho`. Initial scale is 0 (updated by TSPreStep). Pressure Neumann BCs. When applied to `y_dot`, gives `sigma_0 * S * p_dot` for the differentiated continuity.

**Convection** (`fd_conv[d]`):
```
C_d = sum_e d/dx_e(F_e * u_d_TVD)
  where F_e = rho * u_e (mass flux at faces)
```
Using FlucaFD:
1. `FlucaFDSecondOrderTVDCreate(sol_dm, e, d, 0)` — interpolates `u_d` to face using TVD limiter
2. `FlucaFDScaleCreateVector(tvd, mass_flux_face[e], 0)` — multiplies by `F_e`
3. `FlucaFDDerivativeCreate(sol_dm, e, 1, 2, face_loc[e], 0, ELEMENT, d)` — face derivative
4. `FlucaFDCompositionCreate(scaled_tvd, face_deriv)` — compose
5. `FlucaFDSumCreate(dim, conv_comp_per_e)` — sum over e

TVD limiter is configurable via `-phys_ins_tvd_limiter_type`.

**Cell-to-face interpolation** (`fd_interp[d]`):
```
Derivative(ELEMENT, d -> face[d], 0)
```
Used to compute face mass flux from cell-centered velocity.

### Face DMs and mass flux

For each direction `e`:
```c
/* Face DM: 1 DOF at face location */
DMStagCreateCompatibleDMStag(sol_dm, 0, 1 /* 2D edge */, 0, 0, &dm_face[e]);
DMCreateGlobalVector(dm_face[e], &mass_flux_face[e]);
```

### IFunction (implicit part)

```c
/* F_momentum = rho * u_dot - mu * nabla^2(u) + grad(p) */
for (d = 0; d < dim; d++) {
  FlucaFDApply(fd_laplacian[d], t, sol_dm, sol_dm, U, F);   /* -mu * L * u_d -> F[d] */
  FlucaFDApply(fd_grad_p[d], t, sol_dm, sol_dm, U, F);    /* dp/dx_d -> F[d] */
}
/* Add rho * u_dot to momentum rows of F */
AddScaledVelocityDot(rho, Udot, F);

/* F_continuity = alpha * D(u) + D_dot(u_dot) + alpha * sigma_0 * S(p) + sigma_0 * S_dot(p_dot) */
/* Compute constraint feedback: alpha * [D(u) + sigma_0 * S(p)] */
VecZeroEntries(temp);
FlucaFDApply(fd_div, t, sol_dm, sol_dm, U, temp);     /* D(u)           */
FlucaFDApply(fd_pstab, t, sol_dm, sol_dm, U, temp);   /* + sigma_0 * S(p) */
AddScaledContinuity(ins->alpha, temp, F);           /* scale by alpha, add to F[dim] */

/* Compute time derivative part: D_dot(u_dot) + sigma_0 * S_dot(p_dot) */
FlucaFDApplyDot(fd_div, t, sol_dm, sol_dm, Udot, F);   /* D_dot(u_dot)           -> F[dim] */
FlucaFDApplyDot(fd_pstab, t, sol_dm, sol_dm, Udot, F); /* + sigma_0 * S_dot(p_dot) -> F[dim] */
```

### IJacobian

```
[       shift * rho * I - mu * L                G               ]
[                                                                 ]
[ (shift + alpha) * rho * D    (shift + alpha) * sigma_0 * S     ]
```

Assembled using `FlucaFDGetOperator` for each operator:
- Velocity diagonal: `shift * rho` (from `rho * u_dot`)
- Continuity-velocity block: `(shift + alpha) * (rho * D matrix)` (from `alpha * D(u)` + `D_dot(u_dot)`)
- Continuity-pressure block: `(shift + alpha) * (sigma_0 * S matrix)` (from `alpha * sigma_0 * S(p)` + `sigma_0 * S_dot(p_dot)`)

### RHSFunction (explicit part)

```c
/* G_momentum = -C(u) + f(t) */
UpdateConvectionVelocity(U);  /* recompute mass flux and TVD state */
for (d = 0; d < dim; d++) {
  FlucaFDApply(fd_conv[d], t, sol_dm, sol_dm, U, G);  /* -C_d -> G[d] */
}
/* Add body force f(t) if set */
AddBodyForce(phys, t, G);

/* G_continuity = 0 (constraint feedback is in IFunction) */
```

### RHSJacobian

The Picard-linearized convection matrix (frozen velocity). Assembled from `FlucaFDGetOperator` for each `fd_conv[d]`. Only has entries in the velocity-velocity block; pressure and continuity rows are zero.

### PhysSetUpTS_INS

```c
static PetscErrorCode PhysSetUpTS_INS(Phys phys, TS ts)
{
  Phys_INS *ins = (Phys_INS *)phys->data;

  PetscFunctionBegin;
  PetscCall(TSSetDM(ts, phys->sol_dm));

  /* Wire IMEX callbacks */
  PetscCall(TSSetIFunction(ts, NULL, PhysIFunction_INS, phys));
  PetscCall(TSSetIJacobian(ts, ins->J, ins->J, PhysIJacobian_INS, phys));
  PetscCall(TSSetRHSFunction(ts, NULL, PhysRHSFunction_INS, phys));
  PetscCall(TSSetRHSJacobian(ts, ins->J_rhs, ins->J_rhs, PhysRHSJacobian_INS, phys));

  /* Default to TSARKIMEX with stiffly accurate 3rd order scheme */
  PetscCall(TSSetType(ts, TSARKIMEX));

  /* Update sigma_0 and alpha when dt changes */
  PetscCall(TSSetPreStep(ts, PhysPreStep_INS));

  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### TSPreStep callback

```c
static PetscErrorCode PhysPreStep_INS(TS ts)
{
  Phys      phys;
  Phys_INS *ins;
  PetscReal dt;

  PetscFunctionBegin;
  PetscCall(TSGetApplicationContext(ts, &phys));
  ins = (Phys_INS *)phys->data;
  PetscCall(TSGetTimeStep(ts, &dt));

  /* Update sigma_0 = dt / rho and alpha = 1 / dt */
  if (dt != ins->dt_current) {
    PetscCall(FlucaFDScaleSetScalar(ins->fd_pstab, dt / ins->rho));
    ins->alpha      = 1.0 / dt;
    ins->dt_current = dt;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Test

- `tests/phys/ex1.c`: Create INS, verify solution DM has dim+1 element DOFs.
- `tests/phys/ex2.c`: Manufactured Stokes solution (e.g., Taylor-Green at t=0). Set RHSFunction to zero (no convection). Verify `PhysComputeIFunction` produces zero residual for the exact solution with exact `u_dot`.
- `tests/phys/ex3.c`: Verify convection operator with a known rotating flow `u=(y,-x), p=0`. Check that `(u.grad)u` matches the analytical result.

### Tutorial

- `tutorials/phys/ex1.c`: Taylor-Green vortex 2D with TSARKIMEX. Verify L2 errors.
- `tutorials/phys/ex2.c`: Unsteady manufactured solution with body force.

### Verification

```bash
cmake --build build && ctest --test-dir build -R "tests_phys"
```

---

## Phase 3: Additional BC types and preconditioner

### Phase 3a: Additional INS boundary condition types

Add `PHYS_INS_BC_PRESSURE_OUTLET` and `PHYS_INS_BC_SYMMETRY`:

```c
typedef enum {
  PHYS_INS_BC_NONE,
  PHYS_INS_BC_VELOCITY,          /* existing */
  PHYS_INS_BC_PRESSURE_OUTLET,   /* new */
  PHYS_INS_BC_SYMMETRY,          /* new */
} PhysINSBCType;
```

BC-to-FlucaFD translation:

| INS BC type | Velocity operators | Pressure operators |
|-------------|-------------------|--------------------|
| `VELOCITY` | Dirichlet + fn | Neumann (zero) |
| `PRESSURE_OUTLET` | Neumann (zero) | Dirichlet + fn |
| `SYMMETRY` | Dirichlet (normal) / Neumann (tangential) | Neumann (zero) |

When a pressure outlet BC exists, skip creating the pressure null space (`has_pressure_outlet = PETSC_TRUE`).

Test: `tests/phys/ex4.c` — channel flow with velocity inlet + pressure outlet.

### Phase 3b: PCFIELDSPLIT default

Set PCFIELDSPLIT as the default preconditioner for better scalability:

```c
PCFieldSplitSetIS(pc, "velocity", ins->is_vel);
PCFieldSplitSetIS(pc, "pressure", ins->is_p);
PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR);
```

User can override via command line:
```
-pc_fieldsplit_type schur
-fieldsplit_velocity_ksp_type gmres
-fieldsplit_pressure_ksp_type preonly
-fieldsplit_pressure_pc_type jacobi
```

---

## Phase 4: Remove NS module

**Goal**: Delete the entire NS module. Update all dependents.

### Files to delete

| File | Reason |
|------|--------|
| `src/ns/` (entire directory) | Replaced by Phys |
| `src/seg/` (entire directory) | Empty placeholder, never implemented |
| `include/flucans.h` | Replaced by `flucaphys.h` |
| `include/flucansbc.h` | BC types moved to `flucaphys.h` |
| `include/fluca/private/nsimpl.h` | Replaced by `physimpl.h` |
| `include/fluca/private/nslinearcnimpl.h` | Gone |

Remove `add_subdirectory(ns)` and `add_subdirectory(seg)` from `src/CMakeLists.txt`.

### Migrate existing tests and app

| File | Changes |
|------|---------|
| `tests/cavity_flow/cavity_flow_2d.c` | Use DMStagCreate2d + Phys + TS |
| `tests/cavity_flow/cavity_flow_3d.c` | Same for 3D |
| `tests/taylor_green_vortex/taylor_green_vortex.c` | Use Phys + TS |
| `app/main.c` | Rewrite with new API |

Update CMakeLists.txt: link `fluca::phys` instead of `fluca::ns`.

Regenerate golden output files.

### Verification

```bash
cmake --build build && ctest --test-dir build -R "tests_"
```

---

## Key Technical Details

### Base DM vs. Solution DM

The user provides a **base DM** that defines the grid (cell count, boundary types, coordinates, partitioning). The Phys subtype creates a **solution DM** with the correct DOF layout during `PhysSetUp()`:

```
PhysSetBaseDM(phys, base_dm)
    │
    ▼  PhysSetUp(phys)
    │    ├── createsolutiondm op: DMStagCreateCompatibleDMStag(base_dm, ..., &sol_dm)
    │    └── setup op: build FlucaFD operators on sol_dm
    │
PhysGetSolutionDM(phys, &sol_dm)  →  pass to TS
```

For INS: `DMStagCreateCompatibleDMStag(base_dm, 0, 0, 0, dim + 1, &sol_dm)` — dim velocity + 1 pressure at ELEMENT.

### Boundary condition adapter

`PhysINSBCFn` has a `comp` parameter so the user writes one callback for all velocity components. `FlucaFDBCValueFn` on `main` does not have `comp`. The INS subtype bridges this with `PhysINS_BCAdapter`, which carries both `fn` (value) and `fn_dot` (time derivative) callbacks:

```c
typedef struct {
  PhysINSBCFn *fn;       /* value callback */
  PhysINSBCFn *fn_dot;   /* time derivative callback (may be NULL) */
  void        *ctx;
  void        *ctx_dot;
  PetscInt     comp;
} PhysINS_BCAdapter;
```

The adapter provides two `FlucaFDBCValueFn` wrappers: `PhysINS_BCAdapterFn` (for `fn`) and `PhysINS_BCAdapterFnDot` (for `fn_dot`). Both are set on each `FlucaFDBoundaryCondition` entry so that `FlucaFDApply` uses value BCs and `FlucaFDApplyDot` uses time-derivative BCs through the same operator.

The adapter structs are stored in `Phys_INS` as `bc_adapters[comp][face]`. They're initialized once during setup and passed as `fn_ctx` / `fn_dot_ctx` to `FlucaFDSetBoundaryConditions(fd, comp, bcs[])`.

### Null space handling

When all boundaries have velocity Dirichlet BCs (no pressure outlet), pressure is determined only up to a constant. The INS subtype creates an explicit null space vector on the pressure sub-IS:
```c
VecGetSubVector(nullvec, is_p, &subvec);
VecSet(subvec, 1.0 / PetscSqrtReal((PetscReal)np));
VecRestoreSubVector(nullvec, is_p, &subvec);
MatNullSpaceCreate(comm, PETSC_FALSE, 1, &nullvec, &nullspace);
MatSetNullSpace(J, nullspace);
```

### Dimension independence

All operator construction loops over `d = 0..dim-1`. Face location for direction d:
- d=0: `DMSTAG_LEFT`, d=1: `DMSTAG_DOWN`, d=2: `DMSTAG_BACK`

No separate 2D/3D code paths for operator construction. Body force evaluation has dim-specific loops (2D: i,j; 3D: i,j,k) since it accesses coordinate arrays directly.

---

## Files summary

### Modified (Phase 0)

| File | Change |
|------|--------|
| `include/flucafd.h` | Add `t` to `FlucaFDBCValueFn`, `FlucaFDApply`, `FlucaFDGetStencil`; add `fn_dot`, `fn_dot_ctx` to `FlucaFDBoundaryCondition`; declare `FlucaFDApplyDot` |
| `src/fd/interface/fdapply.c` | Thread `t` through `FlucaFDApply`, `FlucaFDGetStencil`, `FlucaFDGetOperator`; implement `FlucaFDApplyDot` |
| `src/fd/utils/fdutils.c` | Thread `t` through `FlucaFDEvaluateBCValue_Internal`, `FlucaFDResolveTVDRefs_Internal` |
| `src/fd/impls/*/` | Propagate dot-BC mode through subtypes |
| `tests/fd/`, `tutorials/fd/` | Update all `FlucaFDApply` and `FlucaFDGetStencil` call sites to include `t` |

### Created (new)

| File | Phase |
|------|-------|
| `include/flucaphys.h` | 1 |
| `include/fluca/private/physimpl.h` | 1 |
| `include/fluca/private/physinsimpl.h` | 2 |
| `src/phys/CMakeLists.txt` | 1 |
| `src/phys/interface/physbasic.c` | 1 |
| `src/phys/interface/physopts.c` | 1 |
| `src/phys/interface/physpkg.c` | 1 |
| `src/phys/interface/physreg.c` | 1 |
| `src/phys/impls/ins/ins.c` | 2 |
| `src/phys/impls/ins/insops.c` | 2 |
| `tests/phys/CMakeLists.txt` | 2 |
| `tests/phys/ex1.c` (DM DOFs) | 2 |
| `tests/phys/ex2.c` (Stokes IFunction) | 2 |
| `tests/phys/ex3.c` (convection) | 2 |
| `tutorials/phys/ex1.c` (Taylor-Green) | 2 |
| `tutorials/phys/ex2.c` (unsteady manufactured) | 2 |

### Deleted (Phase 4)

| File |
|------|
| `src/ns/` (entire directory) |
| `src/seg/` (entire directory) |
| `include/flucans.h` |
| `include/flucansbc.h` |
| `include/fluca/private/nsimpl.h` |
| `include/fluca/private/nslinearcnimpl.h` |

### Migrated (Phase 4)

| File |
|------|
| `tests/cavity_flow/cavity_flow_2d.c` |
| `tests/cavity_flow/cavity_flow_3d.c` |
| `tests/taylor_green_vortex/taylor_green_vortex.c` |
| `app/main.c` |
