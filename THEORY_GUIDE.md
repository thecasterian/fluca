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

Fluca's `Phys` module solves the incompressible equations as a fully coupled system in time. The semi-discretized equations are cast as a differential-algebraic system and advanced with an implicit-explicit (IMEX) Runge-Kutta integrator driving PETSc's `TS` directly. The velocity-pressure coupling is retained and solved to convergence at every step; the classical pressure-velocity decoupling — the fractional step method and SIMPLE — reappears here as a *preconditioner* for the coupled system rather than as the time advancement itself.

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

Fluca applies this factorization as a preconditioner via `PCFIELDSPLIT` of type Schur with full factorization. The two triangular factors are the momentum **predictor** and velocity **corrector** (solves with $\mathbf{A}$), and the middle factor is the **pressure-Poisson** solve (with the Schur complement $\widehat{\mathbf{S}}$) — the classical fractional-step sweep, now used to precondition rather than to time-advance. Perot [3] showed that the fractional step method is itself such an approximate block factorization, and Elman et al. [2] that SIMPLE is another; the choice of the momentum-operator approximation $\widetilde{\mathbf{A}}^{-1}$ in $\widehat{\mathbf{S}}$ selects between them:

- **Fractional step** (default): $\widetilde{\mathbf{A}}^{-1} \approx (\text{shift}\,\rho)^{-1}\mathbf{I}$ retains only the mass/time term, so $\widehat{\mathbf{S}}$ reduces to a scaled pressure Poisson operator. Fluca preconditions the Schur complement with an assembled compact pressure Laplacian, which is spectrally equivalent to $\widehat{\mathbf{S}}$ and covers all pressure modes.
- **SIMPLE**: $\widetilde{\mathbf{A}}^{-1} \approx \operatorname{diag}(\mathbf{A})^{-1}$ gives an assembled approximate Schur complement (`-pc_fieldsplit_schur_precondition selfp`).

The two approximations coincide as $\Delta t \to 0$, where the momentum diagonal is dominated by the mass term. Because the split preconditions an outer Krylov iteration, the fully coupled solution is recovered independently of the choice; only the iteration count differs. The default may be replaced from the options database, so `-pc_fieldsplit_schur_precondition selfp` selects the SIMPLE preconditioner and `-ksp_type preonly -pc_type lu` selects a monolithic direct solve.

## References

1. Y. Zang, R. L. Street, and J. R. Koseff, A non-staggered grid, fractional step method for time-dependent incompressible Navier–Stokes equations in curvilinear coordinates, _J. Comput. Phys._, 114, 18&ndash;33 (1994).
2. H. Elman, V. E. Howle, J. Shadid, R. Shuttleworth, and R. Tuminaro, A taxonomy and comparison of parallel block multi-level preconditioners for the incompressible Navier–Stokes equations, _J. Comput. Phys._, 227, 1790&ndash;1808 (2008).
3. J. Perot, An analysis of the fractional step method, _J. Comput. Phys._, 108, 51&ndash;58 (1993).
4. U. M. Ascher, S. J. Ruuth, and R. J. Spiteri, Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations, _Appl. Numer. Math._, 25, 151&ndash;167 (1997).
