# Theory Guide

This document presents the theoretical foundations and numerical methods implemented in Fluca. It is intended for researchers and developers seeking to understand the mathematical formulation and numerical algorithms underlying the software.

## Table of Contents

- [Governing Equations](#governing-equations)
- [Grid System](#grid-system)
- [Spatial Discretization](#spatial-discretization)
- [Immersed Boundary Method](#immersed-boundary-method)
- [Coupled IMEX Formulation](#coupled-imex-formulation)

## Governing Equations

Fluca is designed to simulate unsteady incompressible viscous flow. Assuming constant density $\rho$, the governing equations consist of the continuity equation (mass conservation) and the Navier-Stokes equations (momentum conservation):

```math
\frac{\partial u_i}{\partial x_i} = 0 \tag{1}
```

```math
\frac{\partial u_i}{\partial t} + \frac{\partial}{\partial x_j} u_i u_j = -\frac{1}{\rho} \frac{\partial p}{\partial x_i} + \frac{\partial}{\partial x_j} \left( \nu \frac{\partial u_i}{\partial x_j} \right) \tag{2}
```

## Grid System

Fluca employs a non-staggered (collocated) grid system, in which all variables, including velocity components and pressure, are stored at cell centers. This arrangement offers several advantages over staggered grids: simplified implementation, more straightforward treatment of boundary conditions, and improved suitability for unstructured or hybrid grids. The collocated arrangement also facilitates extension to three-dimensional domains and complex geometries.

However, collocated grids are susceptible to the checkerboard (odd-even decoupling) instability, which manifests as spurious pressure oscillations in the solution. This phenomenon arises because the pressure gradient at a cell center, computed from neighboring cell-center pressures, is decoupled from the pressure at that cell itself, permitting non-physical pressure modes to persist. To address this issue, Fluca introduces a staggered face-normal velocity component that provides the necessary coupling between the pressure and velocity fields.

The face-normal velocity $U$ is defined as

```math
U = \mathbf{u}_\text{face} \cdot \mathbf{n}
```

where $\mathbf{u}_\text{face}$ denotes the velocity vector evaluated at the face center and $\mathbf{n}$ is the unit outward normal vector of the face. Since each face admits two possible normal directions, a consistent convention must be established. On Cartesian grids, Fluca adopts the convention that the face normal vector is oriented in the positive direction of the coordinate axis perpendicular to the face.

The face velocity is not obtained by simple linear interpolation from neighboring cell centers; rather, it is computed using the Rhie-Chow interpolation scheme, which incorporates a pressure-gradient correction term. This correction establishes proper coupling between the pressure and velocity fields, thereby suppressing the checkerboard instability while preserving the advantages of the collocated grid. In the coupled formulation it appears as the pressure-stabilization operator described in the [Coupled IMEX Formulation](#coupled-imex-formulation) section.

## Spatial Discretization

### Cartesian Grid

Fluca employs the finite difference method (FDM) for spatial discretization on Cartesian grids. This approach is well-suited for structured grids, approximating spatial derivatives by finite differences of function values at discrete grid points.

As an illustrative example, consider the second derivative of a variable $\phi$ in the $x$-direction at cell $(i, j)$, assuming a two-dimensional configuration:

```math
\begin{aligned}
\left. \frac{\partial}{\partial x} \frac{\partial}{\partial x} \phi \right|_{i,j} & \approx \frac{\delta}{\delta x} \left( \frac{\delta}{\delta x} \phi_\text{cell} \right)_\text{face} \\
& = \frac{\left.\frac{\delta}{\delta x}\phi_\text{cell}\right|_{i+1/2,j} - \left.\frac{\delta}{\delta x}\phi_\text{cell}\right|_{i-1/2,j}}{\Delta x_{i-1/2,i+1/2}} \\
& = \frac{(\phi_{i+1,j} - \phi_{i,j}) / \Delta x_{i,i+1} - (\phi_{i,j} - \phi_{i-1,j}) / \Delta x_{i-1,i}}{\Delta x_{i-1/2,i+1/2}}
\end{aligned}
```

Here, $\delta$ denotes a finite difference operator, and subscripts indicate cell indices or face locations (half-integer indices denote face positions). This nested difference formulation ensures consistency with the control volume approach and provides second-order spatial accuracy on uniform grids. On non-uniform grids, second-order accuracy is preserved provided the grid stretching is sufficiently smooth. All spatial derivatives in the governing equations are discretized analogously.

When variable values are required at face locations, they are typically obtained by linear interpolation from neighboring cell centers. For the convective and continuity terms, however, Fluca uses the interpolated face-normal velocity $U$ (with the Rhie-Chow correction), which provides enhanced coupling between the pressure and velocity fields. The discrete divergence of velocity is formed from these face-normal velocities,

```math
\frac{\partial u_i}{\partial x_i} \approx \sum_i \frac{\delta U_i}{\delta x_i}
```

while the convective term is discretized in flux form, transporting each velocity component by the face mass flux. For a two-dimensional Cartesian grid, the discrete divergence of velocity at cell $(i, j)$ is:

```math
\left( \frac{\delta U}{\delta x} + \frac{\delta U}{\delta y} \right)_{i,j} = \frac{U_{i+1/2,j} - U_{i-1/2,j}}{\Delta x_{i-1/2,i+1/2}} + \frac{U_{i,j+1/2} - U_{i,j-1/2}}{\Delta y_{j-1/2,j+1/2}}
```

## Immersed Boundary Method

<!-- TODO: -->

## Coupled IMEX Formulation

Fluca's `Phys` module solves the incompressible equations as a fully coupled system in time. The semi-discretized equations are cast as a differential-algebraic system and advanced with an implicit-explicit (IMEX) Runge-Kutta integrator driving PETSc's `TS` directly. The velocity-pressure coupling is retained and solved to convergence at every step; the classical pressure-velocity decoupling reappears here only as a family of *preconditioners* for the coupled system, never as the time advancement itself.

### Pressure-Stabilized Semi-Discrete System

Discretizing the momentum and continuity equations in space while leaving time continuous, and folding the Rhie-Chow correction into a cell-centered pressure-stabilization operator, yields for the cell-centered velocity $\mathbf{u}$ and pressure $p$:

```math
\rho \frac{d\mathbf{u}}{dt} + \mathbf{N}(\mathbf{u}) + \mathbf{G} p = \mu \mathbf{L} \mathbf{u} + \mathbf{f}(t) \tag{3}
```

```math
\mathbf{D} \mathbf{u} + \sigma_0 \mathbf{S} p = 0 \tag{4}
```

where the discrete operators are

- $\mathbf{N}$: convection, discretized with a second-order TVD scheme built on the face mass flux $\rho \overline{\mathbf{u}}$;
- $\mathbf{G}$: cell-centered pressure gradient;
- $\mathbf{L}$: viscous Laplacian;
- $\mathbf{D} = \rho \, \delta \overline{\mathbf{u}} / \delta x_i$: divergence of the interpolated velocity (carrying the factor $\rho$);
- $\mathbf{S} = \mathbf{L}^\text{wide} - \mathbf{L}^\text{compact}$: pressure stabilization, the difference between the wide (interpolated-gradient) and compact discrete Laplacians. It is the cell-centered form of the collocated Rhie-Chow correction [1], coupling pressure and velocity to suppress the checkerboard modes.

The stabilization coefficient is $\sigma_0 = \Delta t$; since $\mathbf{D}$ already carries $\rho$, the effective coefficient relative to $\nabla \cdot \mathbf{u}$ is $\Delta t / \rho$, the classical Rhie-Chow coefficient [1].

Because the continuity equation (4) contains no time derivative, the system is a **differential-algebraic equation** (DAE). Written as $\mathbf{M}\, d\mathbf{y}/dt = \dots$ with $\mathbf{y} = (\mathbf{u}, p)$, the mass matrix

```math
\mathbf{M} = \begin{bmatrix} \rho \mathbf{I} & 0 \\ 0 & 0 \end{bmatrix}
```

is singular: the pressure is an algebraic variable that instantaneously enforces stabilized incompressibility. The stabilization $\sigma_0 \mathbf{S}$ is what makes this constraint solvable on the collocated grid, by suppressing the checkerboard pressure modes.

### Rhie-Chow Interpolation and Pressure Stabilization

The stabilization operator $\mathbf{S}$ in Eq. (4) originates from the Rhie-Chow interpolation of the face-normal velocity. Interpolating the cell-centered velocity to the faces directly would leave each cell pressure decoupled from its own gradient and admit checkerboard modes; instead the face-normal velocity is constructed from a face-discretized momentum balance, which introduces a pressure-gradient correction:

```math
U = \overline{\mathbf{u}} \cdot \mathbf{n} + \frac{\Delta t}{\rho}\left( \overline{\nabla p} \cdot \mathbf{n} - \left. \frac{\partial p}{\partial n} \right|_\text{face} \right)
```

Here $\overline{(\cdot)}$ denotes linear interpolation from the two adjacent cell centers to the face. The first term is ordinary interpolation; the correction is the difference between the interpolated cell pressure gradient and the compact gradient evaluated directly at the face. Because the compact face gradient couples the two adjacent cell pressures, this correction removes the odd-even decoupling. Following Zang et al. [1], the coefficient is fixed at $\Delta t / \rho$ rather than taken from the momentum-equation diagonal as in the classical Rhie-Chow scheme.

Introducing the interpolation operator $\mathbf{T}$ (cell velocity to face normal), the cell-centered pressure gradient $\mathbf{G}$, and the staggered face-normal pressure gradient $\mathbf{G}^\text{st}$, the face velocity becomes

```math
U = \mathbf{T}\mathbf{u} + \frac{\Delta t}{\rho}\left( \mathbf{T}\mathbf{G} - \mathbf{G}^\text{st} \right) p = \mathbf{T}\mathbf{u} + \frac{\Delta t}{\rho}\,\mathbf{R}\, p, \qquad \mathbf{R} = \mathbf{T}\mathbf{G} - \mathbf{G}^\text{st}
```

where $\mathbf{R}$ is the Rhie-Chow correction operator. The discrete continuity equation is the divergence of these face-normal velocities; writing $\mathbf{D}_0$ for that divergence operator ($\mathbf{D}_0 U = \sum_i \delta U_i / \delta x_i$),

```math
\mathbf{D}_0 \mathbf{T}\mathbf{u} + \frac{\Delta t}{\rho}\, \mathbf{D}_0 \mathbf{R}\, p = 0
```

The two products in $\mathbf{D}_0\mathbf{R} = \mathbf{D}_0\mathbf{T}\mathbf{G} - \mathbf{D}_0\mathbf{G}^\text{st}$ are two discrete Laplacians of the pressure: $\mathbf{D}_0\mathbf{T}\mathbf{G}$ interpolates the cell gradient before differencing, a **wide** ($2\Delta x$-stencil) Laplacian, whereas $\mathbf{D}_0\mathbf{G}^\text{st}$ differences the compact face gradient, a **compact** (nearest-neighbor) Laplacian. Their difference is the stabilization operator:

```math
\mathbf{S} = \mathbf{D}_0\mathbf{R} = \mathbf{L}^\text{wide} - \mathbf{L}^\text{compact}
```

Multiplying the continuity relation by $\rho$ and identifying the divergence of the interpolated velocity $\mathbf{D} = \rho\,\mathbf{D}_0\mathbf{T}$ and $\sigma_0 = \Delta t$ recovers the stabilized constraint (4), $\mathbf{D}\mathbf{u} + \sigma_0 \mathbf{S} p = 0$. The stabilization $\sigma_0 \mathbf{S}$ is therefore the divergence of the Rhie-Chow correction. It vanishes to the order of the truncation error for smooth pressure fields — where the wide and compact Laplacians agree — and acts only on the under-resolved checkerboard modes, which is precisely the coupling required by the collocated grid.

### IMEX Runge-Kutta Advancement

`TSARKIMEX` solves systems of the form $\mathbf{F}(t, \mathbf{y}, \dot{\mathbf{y}}) = \mathbf{G}(t, \mathbf{y})$, where $\mathbf{F}$ collects the stiff terms (advanced implicitly) and $\mathbf{G}$ the non-stiff terms (advanced explicitly) [4]. The terms of (3)-(4) are sorted by stiffness:

```math
\mathbf{F} = \begin{bmatrix} \rho \dot{\mathbf{u}} - \mu \mathbf{L}\mathbf{u} + \mathbf{G} p \\ \mathbf{D}\mathbf{u} + \sigma_0 \mathbf{S} p \end{bmatrix}, \qquad
\mathbf{G} = \begin{bmatrix} -\mathbf{N}(\mathbf{u}) + \mathbf{f}(t) \\ 0 \end{bmatrix}
```

The viscous term (parabolic-stiff), pressure gradient, and the algebraic continuity constraint are integrated implicitly; convection (non-stiff) is integrated explicitly. Keeping convection explicit leaves the implicit residual **linear** in $(\mathbf{u}, p)$, so each implicit stage is a single linear solve with no Newton iteration, at the cost of a convective CFL restriction.

For an $s$-stage scheme with a diagonally-implicit tableau $a^I$, stage $i$ requires an implicit solve whose Jacobian is $\mathbf{J} = \text{shift}\cdot\mathbf{M} + \partial\mathbf{F}/\partial\mathbf{y}$, with the stage shift $\text{shift} = 1/(a^I_{ii}\,\Delta t)$:

```math
\mathbf{J} = \begin{bmatrix} \text{shift}\,\rho \mathbf{I} - \mu \mathbf{L} & \mathbf{G} \\ \mathbf{D} & \sigma_0 \mathbf{S} \end{bmatrix} \tag{5}
```

The singular pressure block of $\mathbf{M}$ contributes nothing to $\mathbf{J}$; the corresponding diagonal is instead supplied by the stabilization $\sigma_0 \mathbf{S}$, so $\mathbf{J}$ is nonsingular apart from the constant-pressure null space (removed by a null-space projection). A **stiffly-accurate** scheme (e.g. `ARKIMEX3`) is required so that the algebraic pressure is advanced at the full order of the method.

### Schur-Complement Preconditioning

The stage matrix (5) is a saddle-point system for the collocated two-field unknowns $(\mathbf{u}, p)$, in which the face-normal velocity has been eliminated and the Rhie-Chow correction absorbed into $\sigma_0 \mathbf{S}$. Writing $\mathbf{A} = \text{shift}\,\rho \mathbf{I} - \mu \mathbf{L}$ and $\mathbf{C} = \sigma_0 \mathbf{S}$, it admits the block LDU factorization

```math
\mathbf{J} =
\begin{bmatrix} \mathbf{I} & 0 \\ \mathbf{D}\mathbf{A}^{-1} & \mathbf{I} \end{bmatrix}
\begin{bmatrix} \mathbf{A} & 0 \\ 0 & \widehat{\mathbf{S}} \end{bmatrix}
\begin{bmatrix} \mathbf{I} & \mathbf{A}^{-1}\mathbf{G} \\ 0 & \mathbf{I} \end{bmatrix},
\qquad \widehat{\mathbf{S}} = \mathbf{C} - \mathbf{D}\mathbf{A}^{-1}\mathbf{G} \tag{6}
```

Fluca applies this factorization as a preconditioner via `PCFIELDSPLIT` of type Schur with full factorization. The two triangular factors are the momentum **predictor** and velocity **corrector**, and the middle factor is the **pressure-Poisson** solve (with the Schur complement $\widehat{\mathbf{S}}$) — the classical fractional-step sweep, now used to precondition rather than to time-advance. Perot [3] showed that the fractional step method is itself such an approximate block factorization, and Elman et al. [2] that SIMPLE is another.

#### Two independent approximations

Following the nomenclature of Quarteroni et al. [5] as adopted by Elman et al. [2], group the lower and diagonal factors of (6) together, $(\mathbf{L}\mathbf{D})\mathbf{U}$:

```math
\mathbf{J} =
\begin{bmatrix} \mathbf{A} & 0 \\ \mathbf{D} & \widehat{\mathbf{S}} \end{bmatrix}
\begin{bmatrix} \mathbf{I} & \mathbf{A}^{-1}\mathbf{G} \\ 0 & \mathbf{I} \end{bmatrix} \tag{7}
```

The $\mathbf{A}^{-1}$ of the lower factor cancels against the diagonal block, so exactly **two** occurrences of $\mathbf{A}^{-1}$ survive, and they are approximated independently:

- $\widetilde{\mathbf{A}}_1$ — the approximation in the **Schur complement**, $\widetilde{\mathbf{S}} = \mathbf{C} - \mathbf{D}\widetilde{\mathbf{A}}_1^{-1}\mathbf{G}$, i.e. in the pressure-Poisson operator;
- $\widetilde{\mathbf{A}}_2$ — the approximation in the **upper triangular factor**, i.e. in the velocity correction $\mathbf{u}^{n+1} = \mathbf{u}^* - \widetilde{\mathbf{A}}_2^{-1}\mathbf{G}p'$.

The resulting preconditioner and its error are

```math
\widetilde{\mathbf{M}} =
\begin{bmatrix} \mathbf{A} & 0 \\ \mathbf{D} & \widetilde{\mathbf{S}} \end{bmatrix}
\begin{bmatrix} \mathbf{I} & \widetilde{\mathbf{A}}_2^{-1}\mathbf{G} \\ 0 & \mathbf{I} \end{bmatrix},
\qquad
\mathbf{E} = \mathbf{J} - \widetilde{\mathbf{M}} =
\begin{bmatrix}
0 & (\mathbf{I} - \mathbf{A}\widetilde{\mathbf{A}}_2^{-1})\mathbf{G} \\
0 & \mathbf{D}(\widetilde{\mathbf{A}}_1^{-1} - \widetilde{\mathbf{A}}_2^{-1})\mathbf{G}
\end{bmatrix} \tag{8}
```

Reading off the two blocks of $\mathbf{E}$:

1. The **momentum** equation is unperturbed iff $\widetilde{\mathbf{A}}_2 = \mathbf{A}$ (*momentum preserving*) — only the pressure gradient seen by the velocity is affected, never the velocity operator itself.
2. The **continuity** equation is unperturbed iff $\widetilde{\mathbf{A}}_1 = \widetilde{\mathbf{A}}_2$ (*mass preserving*) — the two approximations need not be accurate, only **consistent with each other**.

Approximating the Schur complement alone is therefore *not* what defines the fractional step method or SIMPLE. Both are mass-preserving schemes: they use the **same** cheap inverse in the Schur complement and in the velocity correction, and accept a perturbed momentum equation in exchange for an exactly satisfied discrete continuity equation. Keeping $\widetilde{\mathbf{A}}_2 = \mathbf{A}$ while approximating only $\widetilde{\mathbf{A}}_1$ gives a different — momentum-preserving — member of the same family, in which continuity is the perturbed equation.

| Scheme | $\widetilde{\mathbf{A}}_1$ | $\widetilde{\mathbf{A}}_2$ | Unperturbed |
| --- | --- | --- | --- |
| Exact block factorization | $\mathbf{A}$ | $\mathbf{A}$ | both |
| Fractional step [3] | $\text{shift}\,\rho\,\mathbf{I}$ | $\text{shift}\,\rho\,\mathbf{I}$ | continuity |
| SIMPLE [2] | $\operatorname{diag}(\mathbf{A})$ | $\operatorname{diag}(\mathbf{A})$ | continuity |
| Schur-only approximation | $\ne \mathbf{A}$ | $\mathbf{A}$ | momentum |

The fractional-step and SIMPLE choices coincide as $\Delta t \to 0$, where $\operatorname{diag}(\mathbf{A})$ is dominated by the mass term $\text{shift}\,\rho$. Because the split preconditions an outer Krylov iteration, the fully coupled solution is recovered independently of the choice; only the iteration count differs.

#### Mapping onto `PCFIELDSPLIT`

`PCFIELDSPLIT` implements exactly this taxonomy — its internal solver for the upper factor is named after $H_2$ of [2]. The three solves are set independently from the options database:

| Role | Options prefix | Fluca default |
| --- | --- | --- |
| Solve with $\mathbf{A}$ (predictor, lower/diagonal factor) | `-fieldsplit_velocity_` | user-selected |
| $\widetilde{\mathbf{A}}_1$ in $\widetilde{\mathbf{S}}$ | `-fieldsplit_pressure_inner_`, preconditioned via `-pc_fieldsplit_schur_precondition` | falls back to the velocity solve, preconditioned by the assembled `selfp` matrix $\mathbf{S}_p$ |
| $\widetilde{\mathbf{A}}_2$ in the velocity correction | `-fieldsplit_pressure_upper_` | PETSc default: reuses the velocity solve |

Only one prefix appears for the lower and diagonal factors because `PCFIELDSPLIT`, although it documents the `full` factorization as the plain $\mathbf{L}\mathbf{D}\mathbf{U}$ product, applies it in the fused $(\mathbf{L}\mathbf{D})\mathbf{U}$ grouping of (7): a single solve with $\mathbf{A}$ produces both the predicted velocity and the argument of $\mathbf{D}\mathbf{u}^*$ that forms the Schur right-hand side. Each preconditioner application therefore costs **one** solve with $\mathbf{A}$ (the predictor) plus one application of $\widetilde{\mathbf{A}}_2^{-1}$, rather than the three solves a literal $\mathbf{L}\cdot\mathbf{D}\cdot\mathbf{U}$ application would require — and when $\widetilde{\mathbf{A}}_2$ is a cheap approximation rather than a solve with $\mathbf{A}$, only that single predictor solve remains. The cancellation is algebraic only for a fixed linear operator; if the velocity `KSP` is an inexact Krylov method, the two solves with $\mathbf{A}$ do not cancel exactly and the preconditioner is only approximately of the form (8), which is why an outer flexible Krylov method is advisable in that case.

The Schur block enters twice: as the **operator**, applied matrix-free as $\mathbf{C} - \mathbf{D}\mathbf{A}^{-1}\mathbf{G}$ with $\mathbf{A}^{-1}$ supplied by the inner solve (`-fieldsplit_pressure_inner_`, which in PETSc falls back to the velocity solve), and as the **preconditioner** for that operator (`-pc_fieldsplit_schur_precondition`: `user` for a supplied matrix, `selfp` for the assembled $\mathbf{C} - \mathbf{D}\operatorname{diag}(\mathbf{A})^{-1}\mathbf{G}$). If the pressure `KSP` is iterated to a tight tolerance, $\widetilde{\mathbf{A}}_1$ is whatever the *inner* solve applies and the preconditioner only affects the iteration count; if it is `preonly`, the preconditioner matrix alone defines $\widetilde{\mathbf{A}}_1$.

#### Fluca's default

Fluca installs the split itself — `PCFIELDSPLIT` of type Schur with `PC_FIELDSPLIT_SCHUR_FACT_FULL`, the velocity and pressure index sets, and the pressure null space — but approximates neither $\widetilde{\mathbf{A}}_1$ nor $\widetilde{\mathbf{A}}_2$. Both fall back to PETSc's default, the velocity solve, so the default sits in the **first** row of the table above: $\widetilde{\mathbf{A}}_1 = \widetilde{\mathbf{A}}_2 = \mathbf{A}$, perturbing neither momentum nor continuity. All three solves are left to the options database.

What Fluca does set is the **preconditioner matrix** for the Schur block: `PC_FIELDSPLIT_SCHUR_PRE_SELFP`, the explicitly assembled

```math
\mathbf{S}_p = \sigma_0\mathbf{S} - \mathbf{D}\operatorname{diag}(\mathbf{A})^{-1}\mathbf{G}
```

in place of PETSc's default $\mathbf{A}_{11} = \sigma_0\mathbf{S}$. The latter is the stabilization operator alone, which acts on the checkerboard modes and is a weak preconditioner for the smooth ones; $\mathbf{S}_p$ carries the $\mathbf{D}\mathbf{A}^{-1}\mathbf{G}$ term as well and so covers all pressure modes.

This is a choice of preconditioner, not of approximation, and it does **not** make the default SIMPLE. $\widetilde{\mathbf{A}}_1$ is still whatever the *inner* solve applies, so with the pressure `KSP` converged to a tight tolerance the classification above is unchanged and $\mathbf{S}_p$ only lowers the iteration count. Only under `-fieldsplit_pressure_ksp_type preonly` does the preconditioner matrix itself become $\widetilde{\mathbf{A}}_1$, giving $\widetilde{\mathbf{A}}_1 = \operatorname{diag}(\mathbf{A})$ against $\widetilde{\mathbf{A}}_2 = \mathbf{A}$ — the momentum-preserving last row, not SIMPLE. SIMPLE additionally requires $\widetilde{\mathbf{A}}_2 = \operatorname{diag}(\mathbf{A})$ through `-fieldsplit_pressure_upper_`.

Because nothing is baked into the operators, the Jacobian is untouched: $\mathbf{P}_\text{mat} = \mathbf{A}_\text{mat} = \mathbf{J}$, and `-ksp_type preonly -pc_type lu` remains an exact monolithic reference solve. The cheaper members of the family are reachable entirely from the options database — for example SIMPLE:

```
-pc_fieldsplit_schur_precondition selfp
-fieldsplit_pressure_mat_schur_complement_ainv_type diag   # A1 = diag(A) in the assembled S
-fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi
-fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi   # A2 = diag(A)
```

## References

1. Y. Zang, R. L. Street, and J. R. Koseff, A non-staggered grid, fractional step method for time-dependent incompressible Navier–Stokes equations in curvilinear coordinates, _J. Comput. Phys._, 114, 18&ndash;33 (1994).
2. H. Elman, V. E. Howle, J. Shadid, R. Shuttleworth, and R. Tuminaro, A taxonomy and comparison of parallel block multi-level preconditioners for the incompressible Navier–Stokes equations, _J. Comput. Phys._, 227, 1790&ndash;1808 (2008).
3. J. Perot, An analysis of the fractional step method, _J. Comput. Phys._, 108, 51&ndash;58 (1993).
4. U. M. Ascher, S. J. Ruuth, and R. J. Spiteri, Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations, _Appl. Numer. Math._, 25, 151&ndash;167 (1997).
5. A. Quarteroni, F. Saleri, and A. Veneziani, Factorization methods for the numerical approximation of Navier&ndash;Stokes equations, _Comput. Methods Appl. Mech. Engrg._, 188, 505&ndash;526 (2000).
