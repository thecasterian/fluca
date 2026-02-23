---
name: spatial-discretization
description: Reference for DMStag grid layout and FlucaFD finite difference operators. Use when working on spatial discretization, stencils, operator composition, or understanding the collocated grid.
---

# Spatial Discretization: DMStag + FlucaFD

## DMStag Grid Layout

Fluca uses PETSc's **DMStag** for structured grid storage. DOFs live at vertices, edges, faces, and cell centers.

### Stencil Locations

Fluca uses "lower-left" convention — only LEFT/DOWN/BACK faces, never RIGHT/UP/FRONT:

| Location | Position | Typical Variable |
|----------|----------|-----------------|
| `DMSTAG_ELEMENT` | Cell center (x_i, y_j, z_k) | Pressure, scalars |
| `DMSTAG_LEFT` | x-face (x_{i-1/2}) | u velocity |
| `DMSTAG_DOWN` | y-face (y_{j-1/2}) | v velocity |
| `DMSTAG_BACK` | z-face (z_{k-1/2}) | w velocity |

Face count: N+1 (non-periodic) or N (periodic) per direction. Cell centers: always N.

### Coordinates

DMStag product coordinates with two slots per direction:
- `DMSTAG_LEFT` slot → face coordinate x_{i-1/2}
- `DMSTAG_ELEMENT` slot → cell center x_i

Access: `DMStagGetProductCoordinateArraysRead()` + `DMStagGetProductCoordinateLocationSlot()`.

### Global and Local Vectors

DMStag manages two vector types that differ in size and ghost point inclusion.

`entriesPerElement` packs all DOFs at the "lower-left" corner of one element (vertex + edges + faces + element center). For example in 2D with 1 DOF per stratum: `entriesPerElement = 4` (vertex + left-face + down-face + element).

**Global vector** — distributed across all ranks in the MPI communicator, one entry per owned DOF, no ghost points. Uses **global numbering** (indices are unique across all ranks). Size is **not** simply `owned_elements × entriesPerElement` when non-element strata have DOFs and boundaries are non-periodic. The last rank in each direction owns extra DOFs for the upper boundary (right faces, top faces, far-right vertex, etc.) that are not covered by `entriesPerElement`.

**Local vector** — lives on a single process (`PETSC_COMM_SELF`), includes owned DOFs **plus ghost region** (neighbor data needed for stencil operations). Uses **local numbering** (always starts at 0, so index 0 refers to different physical elements on different ranks). Size = `gxm * gym * gzm * entriesPerElement` (the ghost region is always a full rectangle of elements, so `entriesPerElement` applies uniformly).

**Distribution example** — 2D, 2x2 elements, 2 ranks (split in x), stencil width 1, non-periodic, 1 DOF per stratum (`entriesPerElement = 4`: V vertex + D down-face + L left-face + E element):

```
Global vector (numbers are global indices):

    Rank 0    ||         Rank 1
              ||
  V8 -- D9 ---++--- V22 -- D23 -- V24
  |           ||     |             |
  L6    E7    ||    L18    E19    L21
  |           ||     |             |
  V4 -- D5 ---++--- V16 -- D17 -- V20
  |           ||     |             |
  L2    E3    ||    L12    E13    L15
  |           ||     |             |
  V0 -- D1 ---++--- V10 -- D11 -- V14
              ||
  10 entries  ||       15 entries
  (1x2x4 + 2) ||       (1x2x4 + 7)
```

Extra boundary DOFs on the last rank per direction: vertices + the face stratum perpendicular to the boundary. At x=2 (right): V + L (left-face = right boundary face). At y=2 (top): V + D (down-face = top boundary face). At (2,2): V only.

```
Local vectors (numbers are local indices):

Rank 0 (gxs=0, gxm=2, gys=0, gym=3):     24 = 2 x 3 x 4

  L18    E19    L22    E23      ghost(padding)
   |             |
  V16 -- D17 -- V20 -- D21    ^
   |             |            |
  L10    E11    L14    E15    |
   |             |            |
   V8 --- D9 -- V12 -- D13    | owned
   |             |            |
   L2     E3     L6     E7    |
   |             |            |
   V0 --- D1 --- V4 --- D5    v

   <------>      <------>
    owned         ghost

Rank 1 (gxs=0, gxm=3, gys=0, gym=3):     36 = 3 x 3 x 4

  L26    E27    L30    E31    L34    E35      ghost(padding)
   |             |             |
  V24 -- D25 -- V28 -- D29 -- V32 -- D33    ^
   |             |             |            |
  L14    E15    L18    E19    L22    E23    |
   |             |             |            |
  V12 -- D13 -- V16 -- D17 -- V20 -- D21    | owned
   |             |             |            |
   L2     E3     L6     E7    L10    E11    |
   |             |             |            |
   V0 --- D1 --- V4 --- D5 --- V8 --- D9    v

   <------>      <------------->
    ghost             owned         ghost
                                  (padding)
```

Each element occupies 4 sequential slots [V, D, L, E]. At boundary phantom elements, only boundary-facing slots carry real data (V+D at top, V+L at right, V only at corner). The remaining slots are **padding** — they have no physical meaning and exist solely to keep the local vector size exactly `gxm * gym * gzm * entriesPerElement`, so that `arr[j][i][slot]` indexing works uniformly without special-casing boundary elements.

**Typical workflow:**

```c
/* 1. Get a temporary local vector (includes ghost region) */
DMGetLocalVector(dm, &local);

/* 2. Scatter global → local, filling ghost points from neighbor ranks */
DMGlobalToLocal(dm, global, INSERT_VALUES, local);

/* 3. Access as multi-dimensional array: arr[k][j][i][slot] */
DMStagVecGetArrayRead(dm, local, &arr);

/* 4. Loop over owned region (DMStagGetCorners gives start + count + extra) */
DMStagGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm, &xe, &ye, &ze);
for (j = ys; j < ys + ym + ye; j++)
  for (i = xs; i < xs + xm + xe; i++)
    val = arr[j][i][slot];  /* ghost points accessible via neighbor indices */

/* 5. Restore and return */
DMStagVecRestoreArrayRead(dm, local, &arr);
DMRestoreLocalVector(dm, &local);
```

**Key functions:**

| Function | Purpose |
|----------|---------|
| `DMCreateGlobalVector(dm, &g)` | Create a new global vector |
| `DMGetLocalVector(dm, &l)` | Get a temporary local vector (cached, fast) |
| `DMRestoreLocalVector(dm, &l)` | Return a temporary local vector |
| `DMGlobalToLocal(dm, g, mode, l)` | Scatter global → local (fills ghosts) |
| `DMLocalToGlobal(dm, l, mode, g)` | Scatter local → global |
| `DMStagVecGetArray(dm, l, &arr)` | Mutable multi-dim array view (local vec only) |
| `DMStagVecGetArrayRead(dm, l, &arr)` | Read-only array view (local vec only) |
| `DMStagGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm, &xe, &ye, &ze)` | Owned region |
| `DMStagGetGhostCorners(dm, &gxs, &gys, &gzs, &gxm, &gym, &gzm)` | Ghost region (superset of owned) |
| `DMStagGetLocationSlot(dm, loc, c, &slot)` | Array slot index for a specific DOF |

**Corner variable naming:**

| Variable | Meaning | Returned by |
|----------|---------|-------------|
| `xs, ys, zs` | Owned region start indices | `DMStagGetCorners` |
| `xm, ym, zm` | Owned region counts | `DMStagGetCorners` |
| `xe, ye, ze` | Extra counts (0 or 1, for boundary faces) | `DMStagGetCorners` |
| `gxs, gys, gzs` | Ghost region start indices | `DMStagGetGhostCorners` |
| `gxm, gym, gzm` | Ghost region counts | `DMStagGetGhostCorners` |

**Array indexing** — `DMStagVecGetArray` returns an array indexed by global element indices (not zero-based local). In 2D: `arr[j][i][slot]` where `i,j` range over ghost corners (`gxs..gxs+gxm-1`). The last dimension (`slot`) packs all DOFs within one element. Use `DMStagGetLocationSlot()` to find which slot corresponds to a given `(location, component)`.

**xe, ye, ze** — 0 or 1, indicating whether the local rank owns the extra boundary face/vertex in that direction. Loop bounds for face-located DOFs extend by these values.

## FlucaFD Operator System

Polymorphic FD operators that compute stencils mapping input DOFs to output DOFs on DMStag.

### Subtypes

| Type | Purpose | Lazy? |
|------|---------|-------|
| **Derivative** | FD derivative (order p, accuracy q) via Vandermonde | No (pre-computed) |
| **Composition** | outer(inner(x)) — recursive stencil multiply | Yes |
| **Scale** | Multiply by constant or Vec field | Yes |
| **Sum** | f1 + f2 + ... + fn | Yes |
| **SecondOrderTVD** | Flux-limited interpolation (nonlinear) | No |

**Pre-computed** subtypes (Derivative, SecondOrderTVD) build stencil coefficients at `SetUp` time and store them per grid point. **Lazy** subtypes (Composition, Scale, Sum) have no `SetUp` stencil storage — they compute stencils on-the-fly by recursively calling `GetStencilRaw` on their children and combining the results.

### Lifecycle

```
Create → [SetType] → SetFromOptions → SetBoundaryConditions → SetUp → Apply/GetOperator → Destroy
```

Convenience constructors (`FlucaFDDerivativeCreate`, etc.) handle Create + SetType + config.

### Stencil Representation

A stencil = array of `(DMStagStencil col, PetscScalar v)` pairs. The `col.c` field encodes:
- `>= 0`: Interior grid point (component index)
- `FLUCAFD_BOUNDARY_LEFT` (-1) through `_FRONT` (-6): Boundary value marker
- `FLUCAFD_CONSTANT` (-7): Additive constant term

### Core API

```c
FlucaFDGetStencil(fd, i, j, k, &n, col[], v[])   /* Stencil with BC handling */
FlucaFDApply(fd, input_dm, output_dm, x, y)        /* y = fd(x) */
FlucaFDGetOperator(fd, input_dm, output_dm, mat)    /* Sparse matrix (interior only) */
```

### Apply Internal Flow

`FlucaFDApply(fd, input_dm, output_dm, x, y)` computes `y = fd(x)`:

1. **Scatter input**: `DMGlobalToLocal(input_dm, x, INSERT_VALUES, x_local)` — fills ghost points
2. **Zero local output**: `VecZeroEntries(y_local)`
3. **Loop over owned output cells** (range from `DMStagGetCorners` + extra):
   - Call `FlucaFDGetStencil(fd, i, j, k, &ncols, col, v)` — returns stencil with BCs resolved
   - Dot product: for each stencil entry, `col[c].c >= 0` reads from `x_local`, `col[c].c == FLUCAFD_CONSTANT` adds constant, otherwise applies `bcs[].value`
   - Write result to `y_local`
4. **Scatter output**: `DMLocalToGlobal(output_dm, y_local, INSERT_VALUES, y)`

`FlucaFDGetOperator(fd, input_dm, output_dm, mat)` assembles the sparse matrix by the same loop but only inserts interior stencil points (`col[c].c >= 0`) via `MatSetValuesLocal` with `ADD_VALUES`. Boundary and constant terms are excluded. Caller must call `MatAssemblyBegin/End` afterward.

### Boundary Conditions

```c
FlucaFDBoundaryCondition bcs[2 * dim];  /* [left, right, down, up, back, front] */
bcs[0].type = FLUCAFD_BC_DIRICHLET;    /* or _NEUMANN, _NONE */
bcs[0].value = 0.0;
FlucaFDSetBoundaryConditions(fd, bcs);
```

Off-grid stencil points are resolved by `FlucaFDRemoveOffGridPoints_Internal()`:
- **BC_NONE**: Extrapolate from interior (Vandermonde)
- **BC_DIRICHLET**: Include boundary value in polynomial
- **BC_NEUMANN**: Infer from prescribed derivative

### Operator Composition Patterns

**Derivative** (element ↔ face transition):
```c
FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 2,
    DMSTAG_ELEMENT, 0, DMSTAG_LEFT, 0, &ddx);  /* d/dx: elem→face */
```

**Convection** (ρu · ∇φ):
```
TVD(ELEMENT→face) → Scale(ρ) → d/dx(face→ELEMENT)
```
```c
FlucaFDSecondOrderTVDCreate(dm, FLUCAFD_X, 0, 0, &tvd);
FlucaFDScaleCreateConstant(tvd, rho, &scaled);
FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &ddx);
FlucaFDCompositionCreate(scaled, ddx, &conv_x);
```

**Diffusion** (∇·(μ∇φ)):
```
d/dx(ELEMENT→face) → Scale(μ) → d/dx(face→ELEMENT)
```
```c
FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 2, DMSTAG_ELEMENT, 0, DMSTAG_LEFT, 0, &inner);
FlucaFDScaleCreateConstant(inner, mu, &scaled);
FlucaFDDerivativeCreate(dm, FLUCAFD_X, 1, 2, DMSTAG_LEFT, 0, DMSTAG_ELEMENT, 0, &outer);
FlucaFDCompositionCreate(scaled, outer, &diff_x);
```

**Multi-direction sum** (Laplacian, full convection-diffusion):
```c
FlucaFD ops[] = {conv_x, conv_y, neg_diff_x, neg_diff_y};
FlucaFDSumCreate(4, ops, &full);
```

### SecondOrderTVD Usage

Nonlinear — stencil depends on solution. Must update before each Apply:
```c
FlucaFDSecondOrderTVDSetVelocity(tvd, vel, vel_c);
FlucaFDSecondOrderTVDSetCurrentSolution(tvd, phi);
```

11 limiters: superbee, minmod, MC, vanLeer, vanAlbada, Barth-Jespersen, Venkatakrishnan, Koren, upwind, SOU, QUICK.

### Options Prefixes

Multiple operators share the same option names. Use prefixes to disambiguate:
```c
FlucaFDSetOptionsPrefix(fd_x, "x_");  /* -x_flucafd_deriv_order */
FlucaFDSetOptionsPrefix(fd_y, "y_");  /* -y_flucafd_deriv_order */
```

### Key Source Files

| File | Contents |
|------|----------|
| `include/flucafd.h` | Public API |
| `include/fluca/private/flucafdimpl.h` | Internal structs (all subtypes) |
| `src/fd/interface/fdbasic.c` | Create, SetUp, Destroy |
| `src/fd/interface/fdapply.c` | Apply, GetOperator, GetStencil |
| `src/fd/utils/fdutils.c` | Off-grid handling, Vandermonde, DMStag↔DMDA |
| `src/fd/impls/derivative/derivative.c` | Derivative stencil computation |
| `src/fd/impls/composition/composition.c` | Recursive composition |
| `src/fd/impls/scale/scale.c` | Constant/vector scaling |
| `src/fd/impls/sum/sum.c` | Additive combination |
| `src/fd/impls/secondordertvd/secondordertvd.c` | TVD interpolation |
| `src/mesh/impl/cart/cart.c` | DMStag grid creation |
