# Theory Guide

This document presents the theoretical foundations and numerical methods implemented in Fluca. It is intended for researchers and developers seeking to understand the mathematical formulation and numerical algorithms underlying the software.

## Table of Contents

- [Governing Equations](#governing-equations)
- [Grid System](#grid-system)
- [Temporal Discretization](#temporal-discretization)
- [Spatial Discretization](#spatial-discretization)
- [Immersed Boundary Method](#immersed-boundary-method)
- [Matrix Form of the Discrete System](#matrix-form-of-the-discrete-system)
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

The face velocity is not obtained by simple linear interpolation from neighboring cell centers; rather, it is computed using the Rhie-Chow interpolation scheme, which incorporates a pressure-gradient correction term. This correction establishes proper coupling between the pressure and velocity fields, thereby suppressing the checkerboard instability while preserving the advantages of the collocated grid. The details of this interpolation are presented in the temporal discretization section.

## Temporal Discretization

### Navier-Stokes Equation

Fluca employs a second-order accurate time advancement scheme based on the method of Kim & Choi [1]. Second-order temporal accuracy is essential for minimizing numerical dissipation and dispersion errors, which is particularly important for resolving unsteady flow phenomena such as vortex shedding and transition to turbulence. The scheme is unconditionally stable and accommodates both explicit and implicit treatment of different terms.

The discretization begins by applying the Crank-Nicolson scheme to both the convective and viscous terms of the Navier-Stokes equation (2):

```math
\frac{u_i^{n+1} - u_i^n}{\Delta t} + \frac{1}{2} \frac{\partial}{\partial x_j} (u_i^{n+1} u_j^{n+1} + u_i^n u_j^n) = -\frac{1}{\rho} \frac{\partial p^{n+1/2}}{\partial x_i} + \frac{1}{2} \frac{\partial}{\partial x_j} \left( \nu^{n+1} \frac{\partial u_i^{n+1}}{\partial x_j} + \nu^n \frac{\partial u_i^n}{\partial x_j} \right) \tag{3}
```

Here, the pressure is evaluated at the half time step $n+1/2$, consistent with the time-centered nature of the scheme. The product $u_i^{n+1} u_j^{n+1}$ in the convective term introduces strong nonlinearity. To circumvent this difficulty, the convective term is linearized as follows:

```math
u_i^{n+1} u_j^{n+1} + u_i^n u_j^n = u_i^{n+1} u_j^n + u_i^n u_j^{n+1} + O(\Delta t^2) \tag{4}
```

This linearization preserves second-order accuracy in $\Delta t$. The linearized form permits direct solution without inner iterations when viscosity is constant. The resulting fully implicit time advancement scheme is:

```math
\frac{u^{n+1} - u^n}{\Delta t} + \frac{1}{2} \frac{\partial}{\partial x_j} (u_i^{n+1} u_j^n + u_i^n u_j^{n+1}) = -\frac{1}{\rho} \frac{\partial p^{n+1/2}}{\partial x_i} + \frac{1}{2} \frac{\partial}{\partial x_j} \left( \nu^{n+1} \frac{\partial u_i^{n+1}}{\partial x_j} + \nu^n \frac{\partial u_i^n}{\partial x_j} \right) \tag{5}
```

Rearranging to isolate known quantities from unknowns yields:

```math
u_i^{n+1} + \frac{\Delta t}{2} \frac{\partial}{\partial x_j} (u_i^{n+1} u_j^n + u_i^n u_j^{n+1}) - \frac{\Delta t}{2} \frac{\partial}{\partial x_j} \left( \nu^{n+1} \frac{\partial u_i^{n+1}}{\partial x_j} \right) + \frac{\Delta t}{\rho} \frac{\partial p'}{\partial x_i} = u_i^n + \frac{\Delta t}{2} \frac{\partial}{\partial x_j} \left( \nu^n \frac{\partial u_i^n}{\partial x_j} \right) - \frac{\Delta t}{\rho} \frac{\partial q}{\partial x_i} \tag{6}
```

Here, $p'=p^{n+1/2}-q$ denotes the pressure correction and $q$ represents a known pressure field from the previous step. Employing the pressure correction rather than the pressure itself as an unknown enhances accuracy when the governing equation operators are approximated in the solver, since $p'=O(\Delta t)$. The field $q$ is defined as the pressure at the previous half time step, $p^{n-1/2}$, except at the initial time step ($n=0$), where the initial pressure field $p^0$ is used.

The pressure field $p^{n+1}$ is obtained by extrapolation from the known values $p^{n+1/2}$ and $q$. Armfield & Street [3] demonstrated that this extrapolation preserves second-order temporal accuracy for the pressure field.

### Rhie-Chow Interpolation

Based on the temporal discretization presented above, the Rhie-Chow interpolation for the face velocity takes the form:

```math
\mathbf{u}_\text{face}^{n+1} = \overline{\mathbf{u}}^{n+1} + \frac{\Delta t}{\rho} \left[ \overline{\nabla p'} - \left. \nabla p' \right|_\text{face} \right] \tag{7}
```

where $\overline{\phi}$ denotes linear interpolation of $\phi$ from neighboring cell centers to the face. The first term $\overline{\mathbf{u}}^{n+1}$ represents the linear interpolation of cell-centered velocities. The second term is the correction, comprising the difference between the interpolated pressure gradient (from cell centers) and the pressure gradient computed directly at the face. This correction couples the pressure and velocity fields at the face, suppressing checkerboard oscillations.

The coefficient of the correction term is fixed as $\Delta t / \rho$, in contrast to the classical Rhie-Chow interpolation, which employs coefficients derived from the discretized momentum equation. This formulation follows Zang et al. [2], who developed a similar approach for collocated grids.

### Continuity Equation

The continuity equation (1) is enforced at the new time step $n+1$:

```math
\frac{\partial u_i^{n+1}}{\partial x_i} = 0 \tag{8}
```

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

When variable values are required at face locations, they are typically obtained by linear interpolation from neighboring cell centers. However, for the convective and continuity terms, Fluca employs the face-normal velocity $U$ computed via Rhie-Chow interpolation, which provides enhanced coupling between the pressure and velocity fields [1]:

```math
\frac{\partial}{\partial x_j} (u_i^{n+1} u_j^n + u_i^n u_j^{n+1}) \approx \frac{\delta}{\delta x_j} (\overline{u}_i^{n+1} U^n + \overline{u}_i^n \overline{u}_j^{n+1})
```

```math
\frac{\partial u_i^{n+1}}{\partial x_i} \approx \sum_i \frac{\delta U^{n+1}}{\delta x_i}
```

For a two-dimensional Cartesian grid, the discrete divergence of velocity at cell $(i, j)$ is:

```math
\left( \frac{\delta U}{\delta x} + \frac{\delta U}{\delta y} \right)_{i,j} = \frac{U_{i+1/2,j} - U_{i-1/2,j}}{\Delta x_{i-1/2,i+1/2}} + \frac{U_{i,j+1/2} - U_{i,j-1/2}}{\Delta y_{j-1/2,j+1/2}}
```

## Immersed Boundary Method

<!-- TODO: -->

## Matrix Form of the Discrete System

The discretized governing equations are expressed in matrix form using discrete operators. This representation exposes the saddle-point structure of the coupled system and is the basis for the preconditioner analysis that follows. All operators are defined **unscaled** — no $\Delta t$ or $\rho$ factor is folded into them — and this single operator set is used throughout the remainder of the guide.

The spatial operators are:

- $\mathbf{N}(\mathbf{u})$: convection, discretized with a second-order TVD scheme built on the face mass flux $\rho\overline{\mathbf{u}}$;
- $\mathbf{L}$: viscous Laplacian, $(\mathbf{L}\mathbf{u})_i = \delta^2 u_i / \delta x_j \delta x_j$;
- $\mathbf{G}$: cell-centered pressure gradient, $(\mathbf{G}p)_i = \delta p / \delta x_i$;
- $\mathbf{T}$: interpolation of a cell-centered vector to the face normal, $\mathbf{T}\mathbf{v} = \overline{\mathbf{v}} \cdot \mathbf{n}$, where $\overline{(\cdot)}$ denotes linear interpolation from the two adjacent cell centers;
- $\mathbf{G}^\text{st}$: staggered face-normal pressure gradient, evaluated compactly at the face, $\mathbf{G}^\text{st}p = \left. \delta p / \delta n \right|_\text{face}$;
- $\mathbf{D}_0$: divergence of a face-normal field, $\mathbf{D}_0 U = \sum_i \delta U_i / \delta x_i$;
- $\mathbf{D} = \rho\,\mathbf{D}_0\mathbf{T}$: divergence of the interpolated velocity, carrying the factor $\rho$.

### Momentum Equation

Discretizing the momentum equation (2) in space while leaving time continuous gives, for the cell-centered velocity $\mathbf{u}$ and pressure $p$:

```math
\rho \frac{d\mathbf{u}}{dt} + \mathbf{N}(\mathbf{u}) + \mathbf{G} p = \mu \mathbf{L} \mathbf{u} + \mathbf{f}(t) \tag{9}
```

where $\mathbf{f}$ collects body forces and the boundary-condition contributions of the viscous and pressure operators.

### Rhie-Chow Interpolation and Pressure Stabilization

The face-normal velocity is not an independent unknown: it is the derived quantity supplied by the Rhie-Chow interpolation (7). Interpolating the cell-centered velocity to the faces directly would leave each cell pressure decoupled from its own gradient and admit checkerboard modes; the interpolation therefore carries a pressure-gradient correction. In operator form,

```math
U = \mathbf{T}\mathbf{u} + \frac{\Delta t}{\rho}\left( \mathbf{T}\mathbf{G} - \mathbf{G}^\text{st} \right) p = \mathbf{T}\mathbf{u} + \frac{\Delta t}{\rho}\,\mathbf{R}\, p, \qquad \mathbf{R} = \mathbf{T}\mathbf{G} - \mathbf{G}^\text{st} \tag{10}
```

where $\mathbf{R}$ is the Rhie-Chow correction operator: the difference between the interpolated cell pressure gradient and the compact gradient evaluated directly at the face. Because the compact face gradient couples the two adjacent cell pressures, this correction removes the odd-even decoupling. Following Zang et al. [2], the coefficient is fixed at $\Delta t / \rho$ rather than taken from the momentum-equation diagonal as in the classical Rhie-Chow scheme.

The discrete continuity equation is the vanishing divergence of these face-normal velocities, $\mathbf{D}_0 U = 0$, which on substituting (10) becomes

```math
\mathbf{D}_0 \mathbf{T}\mathbf{u} + \frac{\Delta t}{\rho}\, \mathbf{D}_0 \mathbf{R}\, p = 0
```

The two products in $\mathbf{D}_0\mathbf{R} = \mathbf{D}_0\mathbf{T}\mathbf{G} - \mathbf{D}_0\mathbf{G}^\text{st}$ are two discrete Laplacians of the pressure: $\mathbf{D}_0\mathbf{T}\mathbf{G}$ interpolates the cell gradient before differencing, a **wide** ($2\Delta x$-stencil) Laplacian, whereas $\mathbf{D}_0\mathbf{G}^\text{st}$ differences the compact face gradient, a **compact** (nearest-neighbor) Laplacian. Their difference defines the pressure-stabilization operator:

```math
\mathbf{S} = \mathbf{D}_0\mathbf{R} = \mathbf{L}^\text{wide} - \mathbf{L}^\text{compact} \tag{11}
```

Multiplying through by $\rho$ and identifying $\mathbf{D} = \rho\,\mathbf{D}_0\mathbf{T}$ and the stabilization coefficient $\sigma_0 = \Delta t$ gives the stabilized continuity constraint in cell-centered form:

```math
\mathbf{D} \mathbf{u} + \sigma_0 \mathbf{S} p = 0 \tag{12}
```

Since $\mathbf{D}$ already carries $\rho$, the effective coefficient relative to $\nabla \cdot \mathbf{u}$ is $\Delta t / \rho$, the classical Rhie-Chow coefficient [2]. The stabilization $\sigma_0 \mathbf{S}$ is therefore nothing but the divergence of the Rhie-Chow correction, written as a cell-centered pressure operator. It vanishes to the order of the truncation error for smooth pressure fields — where the wide and compact Laplacians agree — and acts only on the under-resolved checkerboard modes, which is precisely the coupling the collocated grid requires. Because $U$ has been eliminated, the discrete system carries only the two cell-centered fields $(\mathbf{u}, p)$.

### Saddle-Point Structure

Any implicit time discretization of (9) turns the velocity terms into a single velocity block $\mathbf{A}$ acting on the new-time velocity, leaving the pressure gradient, the constraint (12), and known right-hand-side data. The result is the two-field saddle-point system

```math
\begin{bmatrix} \mathbf{A} & \mathbf{G} \\ \mathbf{D} & \sigma_0 \mathbf{S} \end{bmatrix}
\begin{bmatrix} \mathbf{u} \\ p \end{bmatrix} =
\begin{bmatrix} \mathbf{r} + \mathbf{b}_\text{mom} \\ b_\text{cont} \end{bmatrix} \tag{13}
```

where $\mathbf{r}$ holds the known terms from previous time levels and $\mathbf{b}_\text{mom}$, $b_\text{cont}$ the constant contributions of the boundary conditions. Only $\mathbf{A}$ depends on the time discretization; the off-diagonal blocks and the stabilized $(2,2)$ block are the same for every scheme. The Crank-Nicolson advancement of the Temporal Discretization chapter gives one such $\mathbf{A}$ — its equations are the momentum balance (9) multiplied through by $\Delta t / \rho$, which is why $\Delta t / \rho$ appears there on the pressure gradient and not here — and the IMEX Runge-Kutta stage operator of the next chapter gives another.

For non-constant viscosity — an eddy viscosity depending on the velocity field, for instance — $\mathbf{A}$ depends on the unknowns and the system must be solved iteratively; with constant viscosity and an explicitly treated convection term it is linear.

The matrix in (13) is large, sparse, and indefinite: the $(2,2)$ block is not the zero block of a classical saddle-point problem, but neither is it definite enough to make the system easy. Direct factorization is prohibitive beyond small grids, so Fluca solves it with a Krylov method preconditioned by an approximate block factorization, developed in the next chapter.

## Coupled IMEX Formulation

Fluca's `Phys` module solves the incompressible equations as a fully coupled system in time. The semi-discrete system (9), (12) is cast as a differential-algebraic system and advanced with an implicit-explicit (IMEX) Runge-Kutta integrator driving PETSc's `TS` directly. The velocity-pressure coupling is retained and solved to convergence at every step; the classical pressure-velocity decoupling reappears here only as a family of _preconditioners_ for the coupled system, never as the time advancement itself.

The discrete operators of the previous chapter are used unchanged, and the stage matrix is the saddle-point system (13). The only new symbols are the Schur complement $\widehat{\mathbf{S}}$ and its approximation $\widetilde{\mathbf{S}}$, both distinct from the pressure-stabilization operator $\mathbf{S}$ of (11).

### Differential-Algebraic Structure

Because the stabilized continuity constraint (12) contains no time derivative, the pair (9), (12) is a **differential-algebraic equation** (DAE). Written as $\mathbf{M}\, d\mathbf{y}/dt = \dots$ with $\mathbf{y} = (\mathbf{u}, p)$, the mass matrix

```math
\mathbf{M} = \begin{bmatrix} \rho \mathbf{I} & 0 \\ 0 & 0 \end{bmatrix}
```

is singular: the pressure is an algebraic variable that instantaneously enforces stabilized incompressibility. The stabilization $\sigma_0 \mathbf{S}$ is what makes this constraint solvable on the collocated grid, by suppressing the checkerboard pressure modes.

### IMEX Runge-Kutta Advancement

`TSARKIMEX` solves systems of the form $\mathbf{F}(t, \mathbf{y}, \dot{\mathbf{y}}) = \mathbf{g}(t, \mathbf{y})$, where $\mathbf{F}$ collects the stiff terms (advanced implicitly) and $\mathbf{g}$ the non-stiff terms (advanced explicitly) [6]. PETSc's documentation calls the explicit part $\mathbf{G}$; it is renamed $\mathbf{g}$ here to avoid a collision with the pressure-gradient operator. The terms of (9) and (12) are sorted by stiffness:

```math
\mathbf{F} = \begin{bmatrix} \rho \dot{\mathbf{u}} - \mu \mathbf{L}\mathbf{u} + \mathbf{G} p \\ \mathbf{D}\mathbf{u} + \sigma_0 \mathbf{S} p \end{bmatrix}, \qquad
\mathbf{g} = \begin{bmatrix} -\mathbf{N}(\mathbf{u}) + \mathbf{f}(t) \\ 0 \end{bmatrix}
```

The viscous term (parabolic-stiff), pressure gradient, and the algebraic continuity constraint are integrated implicitly; convection (non-stiff) is integrated explicitly. Keeping convection explicit leaves the implicit residual **linear** in $(\mathbf{u}, p)$, so each implicit stage is a single linear solve with no Newton iteration, at the cost of a convective CFL restriction.

For an $s$-stage scheme with a diagonally-implicit tableau $a^I$, stage $i$ requires an implicit solve whose Jacobian is $\mathbf{J} = \text{shift}\cdot\mathbf{M} + \partial\mathbf{F}/\partial\mathbf{y}$, with the stage shift $\text{shift} = 1/(a^I_{ii}\,\Delta t)$:

```math
\mathbf{J} = \begin{bmatrix} \text{shift}\,\rho \mathbf{I} - \mu \mathbf{L} & \mathbf{G} \\ \mathbf{D} & \sigma_0 \mathbf{S} \end{bmatrix} \tag{14}
```

The singular pressure block of $\mathbf{M}$ contributes nothing to $\mathbf{J}$; the corresponding diagonal is instead supplied by the stabilization $\sigma_0 \mathbf{S}$, so $\mathbf{J}$ is nonsingular apart from the constant-pressure null space (removed by a null-space projection). A **stiffly-accurate** scheme (e.g. `ARKIMEX3`) is required so that the algebraic pressure is advanced at the full order of the method.

### Schur-Complement Preconditioning

The stage matrix (14) is the saddle-point system (13) with the velocity block $\mathbf{A} = \text{shift}\,\rho \mathbf{I} - \mu \mathbf{L}$ supplied by the IMEX stage. Abbreviating the stabilized $(2,2)$ block as $\mathbf{C} = \sigma_0 \mathbf{S}$, it admits the block LDU factorization

```math
\mathbf{J} =
\begin{bmatrix} \mathbf{I} & 0 \\ \mathbf{D}\mathbf{A}^{-1} & \mathbf{I} \end{bmatrix}
\begin{bmatrix} \mathbf{A} & 0 \\ 0 & \widehat{\mathbf{S}} \end{bmatrix}
\begin{bmatrix} \mathbf{I} & \mathbf{A}^{-1}\mathbf{G} \\ 0 & \mathbf{I} \end{bmatrix},
\qquad \widehat{\mathbf{S}} = \mathbf{C} - \mathbf{D}\mathbf{A}^{-1}\mathbf{G} \tag{15}
```

Fluca applies this factorization as a preconditioner via `PCFIELDSPLIT` of type Schur with full factorization. The two triangular factors are the momentum **predictor** and velocity **corrector**, and the middle factor is the **pressure-Poisson** solve (with the Schur complement $\widehat{\mathbf{S}}$) — the classical fractional-step sweep, now used to precondition rather than to time-advance. Perot [5] showed that the fractional step method is itself such an approximate block factorization, and Elman et al. [4] that SIMPLE is another.

#### Two independent approximations

Following the nomenclature of Quarteroni et al. [7] as adopted by Elman et al. [4], group the lower and diagonal factors of (15) together, (LD)U:

```math
\mathbf{J} =
\begin{bmatrix} \mathbf{A} & 0 \\ \mathbf{D} & \widehat{\mathbf{S}} \end{bmatrix}
\begin{bmatrix} \mathbf{I} & \mathbf{A}^{-1}\mathbf{G} \\ 0 & \mathbf{I} \end{bmatrix} \tag{16}
```

The $\mathbf{A}^{-1}$ of the lower factor cancels against the diagonal block, so exactly **two** occurrences of $\mathbf{A}^{-1}$ survive, and they are approximated independently:

- $\widetilde{\mathbf{A}}_1$ — the approximation in the **Schur complement**, $\widetilde{\mathbf{S}} = \mathbf{C} - \mathbf{D}\widetilde{\mathbf{A}}_1^{-1}\mathbf{G}$, i.e. in the pressure-Poisson operator;
- $\widetilde{\mathbf{A}}_2$ — the approximation in the **upper triangular factor**, i.e. in the velocity correction $\mathbf{u}^{n+1} = \mathbf{u}^* - \widetilde{\mathbf{A}}_2^{-1}\mathbf{G}p'$.

The resulting preconditioner $\widetilde{\mathbf{J}}$ and its error are

```math
\widetilde{\mathbf{J}} =
\begin{bmatrix} \mathbf{A} & 0 \\ \mathbf{D} & \widetilde{\mathbf{S}} \end{bmatrix}
\begin{bmatrix} \mathbf{I} & \widetilde{\mathbf{A}}_2^{-1}\mathbf{G} \\ 0 & \mathbf{I} \end{bmatrix},
\qquad
\mathbf{E} = \mathbf{J} - \widetilde{\mathbf{J}} =
\begin{bmatrix}
0 & (\mathbf{I} - \mathbf{A}\widetilde{\mathbf{A}}_2^{-1})\mathbf{G} \\
0 & \mathbf{D}(\widetilde{\mathbf{A}}_1^{-1} - \widetilde{\mathbf{A}}_2^{-1})\mathbf{G}
\end{bmatrix} \tag{17}
```

Reading off the two blocks of $\mathbf{E}$:

1. The **momentum** equation is unperturbed iff $\widetilde{\mathbf{A}}_2 = \mathbf{A}$ (_momentum preserving_) — only the pressure gradient seen by the velocity is affected, never the velocity operator itself.
2. The **continuity** equation is unperturbed iff $\widetilde{\mathbf{A}}_1 = \widetilde{\mathbf{A}}_2$ (_mass preserving_) — the two approximations need not be accurate, only **consistent with each other**.

Approximating the Schur complement alone is therefore _not_ what defines the fractional step method or SIMPLE. Both are mass-preserving schemes: they use the **same** cheap inverse in the Schur complement and in the velocity correction, and accept a perturbed momentum equation in exchange for an exactly satisfied discrete continuity equation. Keeping $\widetilde{\mathbf{A}}_2 = \mathbf{A}$ while approximating only $\widetilde{\mathbf{A}}_1$ gives a different — momentum-preserving — member of the same family, in which continuity is the perturbed equation.

| Scheme | $\widetilde{\mathbf{A}}_1$ | $\widetilde{\mathbf{A}}_2$ | Unperturbed |
| --- | --- | --- | --- |
| Exact block factorization | $\mathbf{A}$ | $\mathbf{A}$ | both |
| Fractional step [5] | $\text{shift}\,\rho\,\mathbf{I}$ | $\text{shift}\,\rho\,\mathbf{I}$ | continuity |
| SIMPLE [4] | $\operatorname{diag}(\mathbf{A})$ | $\operatorname{diag}(\mathbf{A})$ | continuity |
| Schur-only approximation | $\ne \mathbf{A}$ | $\mathbf{A}$ | momentum |

The fractional-step and SIMPLE choices coincide as $\Delta t \to 0$, where $\operatorname{diag}(\mathbf{A})$ is dominated by the mass term $\text{shift}\,\rho$. Because the split preconditions an outer Krylov iteration, the fully coupled solution is recovered independently of the choice; only the iteration count differs.

#### Mapping onto `PCFIELDSPLIT`

`PCFIELDSPLIT` implements exactly this taxonomy — its internal solver for the upper factor is named after $H_2$ of [4]. The solves and the Schur preconditioner matrix are set independently from the options database:

| Role | Options prefix | Fluca default |
| --- | --- | --- |
| Solve with $\mathbf{A}$ (predictor, lower/diagonal factor) | `-fieldsplit_velocity_` | user-selected |
| $\widetilde{\mathbf{A}}_1$ in $\widetilde{\mathbf{S}}$ | `-fieldsplit_pressure_inner_` | PETSc default: falls back to the velocity solve |
| Preconditioner matrix for the Schur block | `-pc_fieldsplit_schur_precondition` | PETSc default: $\mathbf{A}_{11} = \sigma_0\mathbf{S}$ |
| $\widetilde{\mathbf{A}}_2$ in the velocity correction | `-fieldsplit_pressure_upper_` | PETSc default: reuses the velocity solve |

Only one prefix appears for the lower and diagonal factors because `PCFIELDSPLIT`, although it documents the `full` factorization as the plain LDU product, applies it in the fused (LD)U grouping of (16): a single solve with $\mathbf{A}$ produces both the predicted velocity and the argument of $\mathbf{D}\mathbf{u}^*$ that forms the Schur right-hand side. Each preconditioner application therefore costs **two** solves with $\mathbf{A}$ — the predictor and the velocity correction — rather than the three a literal L·D·U application would require.

Note that the second solve does not disappear when $\widetilde{\mathbf{A}}_2$ is a cheap approximation. When a separate upper solver is configured (`kspUpper != kspA` in `PCApply_FieldSplit_Schur`), PETSc recomputes the predictor with $\mathbf{A}$ before applying $\widetilde{\mathbf{A}}_2^{-1}$, so the exact $\mathbf{A}$-solve count stays at two and only the *correction* becomes cheap. The saving from a cheap $\widetilde{\mathbf{A}}_2$ is in the correction, not in the predictor count.

The cancellation that produces the fused grouping is algebraic only for a fixed linear operator; if the velocity `KSP` is an inexact Krylov method, the two solves with $\mathbf{A}$ do not cancel exactly and the preconditioner is only approximately of the form (17), which is why an outer flexible Krylov method is advisable in that case.

The Schur block enters twice: as the **operator**, applied matrix-free as $\mathbf{C} - \mathbf{D}\mathbf{A}^{-1}\mathbf{G}$ with $\mathbf{A}^{-1}$ supplied by the inner solve (`-fieldsplit_pressure_inner_`, which in PETSc falls back to the velocity solve), and as the **preconditioner** for that operator (`-pc_fieldsplit_schur_precondition`: `a11` for the $(1,1)$ block itself, `user` for a supplied matrix, `selfp` for the assembled $\mathbf{C} - \mathbf{D}\operatorname{diag}(\mathbf{A})^{-1}\mathbf{G}$). If the pressure `KSP` is iterated to a tight tolerance, $\widetilde{\mathbf{A}}_1$ is whatever the _inner_ solve applies and the preconditioner only affects the iteration count; if it is `preonly`, the preconditioner matrix alone defines $\widetilde{\mathbf{A}}_1$.

One PETSc behavior is worth knowing when relying on the fallback: if no separate inner solver is configured, `PCFIELDSPLIT` sets the velocity `KSP` type to `gmres`, overriding the `preonly` that `PCFieldSplitSetIS` installs, so that the $\mathbf{A}^{-1}$ inside the matrix-free Schur complement is an accurate solve rather than a single preconditioner application. This happens before the velocity `KSP` reads its own options, so `-fieldsplit_velocity_ksp_type` still takes precedence — but leaving it unset does not give a `preonly` velocity solve.

#### Fluca's default

Fluca installs the split itself — `PCFIELDSPLIT` of type Schur with `PC_FIELDSPLIT_SCHUR_FACT_FULL`, the velocity and pressure index sets, and the pressure null space — but approximates neither $\widetilde{\mathbf{A}}_1$ nor $\widetilde{\mathbf{A}}_2$. Both fall back to PETSc's default, the velocity solve, so the default sits in the **first** row of the table above: $\widetilde{\mathbf{A}}_1 = \widetilde{\mathbf{A}}_2 = \mathbf{A}$, perturbing neither momentum nor continuity. All three solves, and the choice of Schur preconditioner matrix, are left to the options database.

Because the split is installed before `TSSetFromOptions` runs, `PCFIELDSPLIT` already has its composite type set to Schur when `PCSetFromOptions` executes, so the `-pc_fieldsplit_schur_*` options are read normally and no special replay is needed.

The Schur preconditioner matrix is therefore PETSc's default, $\mathbf{A}_{11} = \sigma_0\mathbf{S}$. This is the stabilization operator alone, which acts on the checkerboard modes and is a weak preconditioner for the smooth ones. The `selfp` alternative, the explicitly assembled

```math
\mathbf{S}_p = \sigma_0\mathbf{S} - \mathbf{D}\operatorname{diag}(\mathbf{A})^{-1}\mathbf{G}
```

carries the $\mathbf{D}\mathbf{A}^{-1}\mathbf{G}$ term as well and so covers all pressure modes; it is selected with `-pc_fieldsplit_schur_precondition selfp`.

Note that `selfp` is a choice of preconditioner, not of approximation, and it does **not** by itself make the solver SIMPLE. $\widetilde{\mathbf{A}}_1$ is still whatever the _inner_ solve applies, so with the pressure `KSP` converged to a tight tolerance the classification above is unchanged and $\mathbf{S}_p$ only lowers the iteration count. Only under `-fieldsplit_pressure_ksp_type preonly` does the preconditioner matrix itself become $\widetilde{\mathbf{A}}_1$, giving $\widetilde{\mathbf{A}}_1 = \operatorname{diag}(\mathbf{A})$ against $\widetilde{\mathbf{A}}_2 = \mathbf{A}$ — the momentum-preserving last row, not SIMPLE. SIMPLE additionally requires $\widetilde{\mathbf{A}}_2 = \operatorname{diag}(\mathbf{A})$ through `-fieldsplit_pressure_upper_`.

Because nothing is baked into the operators, the Jacobian is untouched: $\mathbf{P}_\text{mat} = \mathbf{A}_\text{mat} = \mathbf{J}$, and `-ksp_type preonly -pc_type lu -pc_factor_shift_type nonzero` remains an exact monolithic reference solve. The shift is required: the $(2,2)$ block is $\sigma_0\mathbf{S}$, which is singular on the constant-pressure mode, so an unshifted `LU` hits a zero pivot and the step fails. The cheaper members of the family are reachable entirely from the options database — for example SIMPLE:

```
-pc_fieldsplit_schur_precondition selfp
-fieldsplit_pressure_mat_schur_complement_ainv_type diag   # A1 = diag(A) in the assembled S
-fieldsplit_pressure_inner_ksp_type preonly -fieldsplit_pressure_inner_pc_type jacobi
-fieldsplit_pressure_upper_ksp_type preonly -fieldsplit_pressure_upper_pc_type jacobi   # A2 = diag(A)
```

## References

1. D. Kim and H. Choi, A Second-Order Time-Accurate Finite Volume Method for Unsteady Incompressible Flow on Hybrid Unstructured Grids, _J. Comput. Phys._, 162, 411&ndash;428 (2000).
2. Y. Zang, R. L. Street, and J. R. Koseff, A non-staggered grid, fractional step method for time-dependent incompressible Navier–Stokes equations in curvilinear coordinates, _J. Comput. Phys._, 114, 18&ndash;33 (1994).
3. S. Armfield and R. Street, The pressure accuracy of fractional-step methods for the Navier-Stokes equations on staggered grids, _ANZIAM J._, 44, C20&ndash;C39 (2003).
4. H. Elman, V. E. Howle, J. Shadid, R. Shuttleworth, and R. Tuminaro, A taxonomy and comparison of parallel block multi-level preconditioners for the incompressible Navier–Stokes equations, _J. Comput. Phys._, 227, 1790&ndash;1808 (2008)
5. J. Perot, An analysis of the fractional step method, _J. Comput. Phys._, 108, 51&ndash;58 (1993).
6. U. M. Ascher, S. J. Ruuth, and R. J. Spiteri, Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations, _Appl. Numer. Math._, 25, 151&ndash;167 (1997).
7. A. Quarteroni, F. Saleri, and A. Veneziani, Factorization methods for the numerical approximation of Navier&ndash;Stokes equations, _Comput. Methods Appl. Mech. Engrg._, 188, 505&ndash;526 (2000).
