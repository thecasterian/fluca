# Theory Guide

This document presents the theoretical foundations and numerical methods implemented in Fluca. It is intended for researchers and developers seeking to understand the mathematical formulation and numerical algorithms underlying the software.

## Table of Contents

- [Governing Equations](#governing-equations)
- [Grid System](#grid-system)
- [Spatial Discretization](#spatial-discretization)
- [Temporal Discretization](#temporal-discretization)
- [References](#references)

## Governing Equations

Fluca simulates unsteady incompressible viscous flow. Assuming constant density $\rho$, the governing equations consist of the continuity equation (mass conservation) and the Navier-Stokes equations (momentum conservation):

```math
\frac{\partial u_i}{\partial x_i} = 0 \tag{1}
```

```math
\frac{\partial u_i}{\partial t} + \frac{\partial}{\partial x_j} u_i u_j = -\frac{1}{\rho} \frac{\partial p}{\partial x_i} + \frac{\partial}{\partial x_j} \left( \nu \frac{\partial u_i}{\partial x_j} \right) \tag{2}
```

## Grid System

Fluca employs a non-staggered (collocated) grid system, in which all variables, including velocity components and pressure, are stored at cell centers. This arrangement offers several advantages over staggered grids: simplified implementation, more straightforward treatment of boundary conditions, and improved suitability for unstructured or hybrid grids.

However, collocated grids are susceptible to the checkerboard (odd-even decoupling) instability, which manifests as spurious pressure oscillations. This arises because the pressure gradient at a cell center, computed from neighboring cell-center pressures, is decoupled from the pressure at that cell itself. To address this, Fluca introduces pressure stabilization.

In the notation of Bakhvalov [1], the semidiscrete system on the collocated grid takes the form (assuming $\sigma_1 = 0$ and $H(t) = 0$):

```math
\frac{\partial u}{\partial t} + F_\text{conv}(u) + Gp = F_\text{diff}(u) + F_\text{source}(t) \tag{3}
```

```math
Du + \sigma_0 Sp = 0 \tag{4}
```

where

- $F_\text{conv}(u) = \nabla \cdot (u \otimes u)$ is the convective (nonlinear advection) term,
- $F_\text{diff}(u) = \nabla \cdot (\nu \nabla u)$ is the viscous diffusion term,
- $F_\text{source}(t)$ is an external body force per unit mass (e.g., gravity),
- $G$ is the discrete gradient scaled by $1/\rho$ and $D$ is the discrete divergence scaled by $\rho$,

and $S = DG - L$ is the pressure stabilization operator. Here $L$ is the discrete (compact) Laplacian (direct second derivative at cell centers, 3-point stencil per direction), while $DG$ is the discrete wide Laplacian (composition of two cell-centered first derivatives, 5-point stencil per direction). The wide stencil skips every other cell center, which is exactly the decoupling that permits checkerboard modes; the difference $S$ measures this decoupling and provides pressure stabilization. The choice of $\sigma_0$ is well explained in [1], and Fluca uses $\sigma_0 = \hat{\tau}$ (see the temporal discretization section for its definition).

An important assumption underlies this formulation: equation (3) assumes that the discretization of $F_\text{conv}$ does not involve pressure. In Fluca's collocated grid, the Rhie-Chow interpolation reconstructs face velocities using pressure differences, so pressure does enter the convective flux. The paper [1] notes that bypassing the high-order treatment of pressure in the convection term does not compromise the overall accuracy, but this interaction has not been studied in detail.

## Spatial Discretization

### Finite Difference Operators (FlucaFD)

Fluca discretizes spatial derivatives using the `FlucaFD` operator framework, which provides composable finite difference operators on PETSc `DMStag`. Each operator maps values at one grid location (e.g., cell centers) to another (e.g., face centers), enabling the construction of complex discretizations from simple primitives.

#### Derivative Operator

The basic building block. Given a direction $d$, derivative order $n$, and accuracy order $k$, the `derivative` operator computes an $n$-th order finite difference in direction $d$ with $k$-th order accuracy. For example, a first derivative in $x$ from element centers to left faces:

```math
\left. \frac{\delta \phi}{\delta x} \right|_{i+1/2,j} = \frac{\phi_{i+1,j} - \phi_{i,j}}{\Delta x_{i,i+1}}
```

Boundary conditions (Dirichlet or Neumann) are applied at the stencil level, modifying coefficients near domain boundaries.

#### Composition Operator

The `composition` operator composes two operators $A \circ B$, applying $B$ first and then $A$. This naturally produces a compact second derivative stencil:

```math
\frac{\partial^2}{\partial x^2} \approx \left( \frac{\delta}{\delta x} \right)_\mathrm{face \to elem} \circ \left( \frac{\delta}{\delta x} \right)_\mathrm{elem \to face}
```

The Laplacian in $d$ dimensions is then the sum of $d$ such compositions.

#### Scale Operator

The `scale` operator multiplies an operator's output by a constant scalar or a spatially-varying field stored in a `Vec`. This is used to incorporate variable coefficients such as viscosity or density into operators.

#### Sum Operator

The `sum` operator adds the outputs of multiple operators. Combined with composition and scaling, this enables construction of the full discrete Laplacian, divergence, and gradient operators from primitive derivatives.

#### Second-Order TVD Operator

The `secondordertvd` operator implements second-order Total Variation Diminishing (TVD) convection schemes for the nonlinear convective term $F_\text{conv}$. It interpolates a cell-centered scalar $\phi$ to a face using a flux-limited combination of the two neighboring cell values:

```math
\phi_{i+1/2} = \frac{1}{2}(\phi_{i} + \phi_{i+1}) + \frac{1}{2}\left[\psi(r^+)\,\alpha^+\,(\phi_{i+1} - \phi_i) + \psi(r^-)\,\alpha^-\,(\phi_{i} - \phi_{i-1})\right]
```

where $r^{\pm}$ are the ratios of consecutive gradients (smoothness monitors), $\psi$ is the flux limiter function, and $\alpha^{\pm}$ are upwind/downwind weights determined by the sign of the mass flux at the face. When the upstream face lies outside the domain, the operator falls back to plain averaging.

The TVD operator is used within the convection discretization as follows: for each velocity component $u_d$, the face value is interpolated via the TVD scheme, multiplied by the mass flux $F_e = \rho\, \overline{u}_e$ at that face, and then differentiated back to cell centers. The full convective operator for component $d$ is:

```math
F_{\text{conv},d}(u) = \sum_e \frac{\delta}{\delta x_e}\left(F_e \cdot \text{TVD}_e(u_d)\right)
```

Available flux limiters (selectable via `-flucafd_limiter <name>`):

| Name | Second-order TVD | Notes |
|------|-----|-------|
| `superbee` | yes | Most compressive; preferred near discontinuities but can over-compress smooth gradients |
| `minmod` | yes | Most diffusive TVD limiter; very stable but smears sharp features |
| `mc` | yes | Monotonized central; good balance between accuracy and compressiveness |
| `vanleer` | yes | Smooth (differentiable) limiter; good general-purpose choice |
| `vanalbada` | yes | Smooth limiter similar to van Leer; slightly less compressive |
| `barthjesperson` | yes | Designed for unstructured grids; symmetric and smooth |
| `venkatakrishnan` | yes | Modified Barth-Jesperson; avoids limiter stalling near smooth extrema |
| `koren` | yes | Third-order accurate on uniform grids; good for smooth flows |
| `upwind` | no | First-order upwind ($\psi = 0$); maximally diffusive, use as baseline |
| `sou` | no | Second-order upwind; unbounded, may produce oscillations |
| `quick` | no | QUICK scheme; unbounded, third-order on uniform grids |

Limiters not marked as second-order TVD are provided for reference and comparison purposes only; they do not satisfy the second-order TVD property and may produce oscillations near sharp gradients or a less accurate solution.

### Discrete Operators for the Navier-Stokes Equations

Using the FlucaFD primitives, `PhysINS` constructs the following discrete operators during setup:

- **Velocity Divergence** $D$: $\rho \sum_d \delta u_d / \delta x_d$. Sum of cell-centered first derivatives, scaled by $\rho$.
- **Pressure gradient** $G_d$: $(1/\rho) \cdot \delta p / \delta x_d$. First derivative of pressure in direction $d$, cell center to cell center, scaled by $1/\rho$. One operator per direction $d$.
- **Pressure Laplacian** $L$: $\sum_d \delta^2 p / \delta x_d^2$. Compact (direct 3-point per direction) second derivative of pressure at cell centers.
- **Convection** $F_{\text{conv},d}$: $\sum_e (\delta/\delta x_e)(F_e \cdot \text{TVD}_e(u_d))$ where $F_e = \rho\,\overline{u}_e$ is the mass flux at face $e$. Uses the second-order TVD operator for face interpolation. One operator per velocity component $d$. Computes momentum flux divergence $[\nabla \cdot (\rho u \otimes u)_d]$, which includes $\rho$ through the mass flux.
- **Viscous diffusion** $F_{\text{diff},d}$: $\sum_e (\delta/\delta x_e) (\nu \cdot \delta/\delta x_e)$ applied to $u_d$, where $\nu = \mu/\rho$ is the kinematic viscosity. Composition of two first derivatives in each direction $e$. One operator per velocity component $d$.

## Temporal Discretization

### Segregated Runge-Kutta Method

Fluca uses the method of Bakhvalov [1] for time integration, applying an IMEX (implicit-explicit) Runge-Kutta scheme to the incompressible Navier-Stokes equations. Viscous diffusion is treated implicitly and convection plus the pressure gradient explicitly, so that each stage decouples into independent Helmholtz (velocity) and Poisson (pressure) solves.

### From Index-2 DAE to ODE

The semidiscrete system (3)-(4) is a differential-algebraic equation (DAE) of index 2: the continuity equation (4) is an algebraic constraint on $u$ and $p$ with no time derivative. Standard ODE integrators cannot be applied directly to an index-2 DAE.

Following [1, Section 5.2], the constraint (4) is differentiated in time and combined with the original to form an ODE:

```math
\alpha \left( Du + \sigma_0 Sp \right) + \frac{\partial}{\partial t}\left( Du + \sigma_0 Sp \right) = 0 \tag{5}
```

where $\alpha > 0$ is the Baumgarte stabilization parameter. This is the Baumgarte method: by adding $\alpha$ times the original constraint to its time derivative, the constraint becomes asymptotically stable rather than merely conserved. If the constraint is satisfied at $t = 0$, both the original constraint and its time derivative are enforced; if not, the error decays exponentially with rate $\alpha$.

Expanding the time derivative in (5) and substituting $\partial u/\partial t$ from the momentum equation (3):

```math
\alpha \left( Du + \sigma_0 Sp \right) + D\left( -F_\text{conv} - Gp + F_\text{diff} + F_\text{source} \right) + \sigma_0 S \frac{\partial p}{\partial t} = 0
```

Denoting $\Xi = Du + \sigma_0 Sp$ (the continuity residual) and $R = -F_\text{conv} - Gp + F_\text{diff} + F_\text{source}$ (the momentum right-hand side), this determines $\partial p/\partial t$, completing the ODE system for $y = (u, p)^T$:

```math
\frac{\partial y}{\partial t} = \xi(t,y) + \eta(t,y) \tag{6}
```

where

```math
\xi = \begin{pmatrix} -F_\text{conv} - Gp + F_\text{source} \\ 0 \end{pmatrix}
```

is treated explicitly and

```math
\eta = \begin{pmatrix} F_\text{diff} \\ -(\sigma_0 S)^{-1}(DR + \alpha \Xi) \end{pmatrix}
```

is treated implicitly.

### IMEX Runge-Kutta Methods

An IMEX (implicit-explicit) Runge-Kutta method advances an ODE of the form $\partial y / \partial t = \xi(t,y) + \eta(t,y)$ by treating $\xi$ explicitly and $\eta$ implicitly. An $s$-stage IMEX RK scheme is defined by two Butcher tableaux:

- $(\hat{A}, \hat{b})$ for the **explicit** part, where $\hat{A}$ is strictly lower triangular (each stage uses only previously computed values),
- $(A, b)$ for the **implicit** part, where $A$ is lower triangular with nonzero diagonal (each stage requires an implicit solve).

At each stage $j = 1, \ldots, s$, the method computes stage values $Y_j$ and **stage slopes**:

- $\hat{K}_j = \xi(t_j, Y_j)$: the explicit slope (evaluation of $\xi$ at the stage value),
- $K_j = \eta(t_j, Y_j)$: the implicit slope (evaluation of $\eta$ at the stage value).

The stage value is determined by:

```math
Y_j = y^{n-1} + \tau \sum_{k=1}^{s} \left( a_{jk}\, K_k + \hat{a}_{jk}\, \hat{K}_k \right)
```

where $\tau$ is the time step size. Since $\hat{A}$ is strictly lower triangular, $\hat{K}_j$ depends only on $Y_j$ (already known). Since $A$ is lower triangular, the sum over $K_k$ includes $k = j$, yielding an implicit equation for $Y_j$. The solution is then updated as $y^n = y^{n-1} + \tau \sum_k (b_k K_k + \hat{b}_k \hat{K}_k)$.

IMEX RK schemes fall into two structural classes based on the implicit tableau $A$:

- **CK type** (Carpenter, Kennedy): The general form, requiring only that $A$ is lower triangular with $a_{11} = 0$ (the first stage is explicit) and $a_{jj} \neq 0$ for $j \geq 2$ (subsequent stages are implicit). The first column of $A$ may be nonzero ($a_{j1} \neq 0$ for some $j > 1$), coupling the first (explicit) stage into subsequent implicit stages. This allows greater design freedom for higher-order accuracy and stability.
- **ARS type** (Ascher, Ruuth, Spiteri): A special case of CK where the first column of $A$ is entirely zero ($a_{j1} = 0$ for all $j$), so the implicit part is fully decoupled from the first stage.

A scheme is **stiffly accurate (SA)** if the last row of the implicit tableau equals the weight vector ($b = A_{s,:}$). All IMEX RK schemes used in Fluca satisfy $b = \hat{b}$ (the implicit and explicit weight vectors are equal), so the solution update simplifies to $y^n = Y_s$ (the last stage value).

### IMEX Split for Incompressible Navier-Stokes

The IMEX split (6) assigns:

- **Explicit** $\xi$: convection $F_\text{conv}$, pressure gradient $Gp$, and source $F_\text{source}$
- **Implicit** $\eta$: viscous diffusion $F_\text{diff}$ and the pressure constraint $\Xi$

The crucial point is that the pressure gradient $Gp$ is in the **explicit** part. The implicit solve at each stage involves only the viscous Helmholtz equation for velocity, with no pressure unknown; the pressure is then determined separately by a Poisson-type equation derived from the constraint. In a conventional coupled IMEX approach, $Gp$ would appear in the implicit part, coupling velocity and pressure into a saddle-point system at each stage. This decoupling is why Bakhvalov [1] calls the method **Segregated Runge-Kutta (SRK)**.

For the velocity components, the stage slopes are:

- $K_j^u = F_\text{diff}(u_j)$: implicit velocity slope (viscous diffusion per unit mass),
- $\hat{K}_j^u = C_j^u - Gp_j$: explicit velocity slope, where $C_j^u = -F_\text{conv}(u_j) + F_\text{source}(t_j)$.

The SRK method requires the implicit tableau to have a constant SDIRK diagonal $a_{jj} = \gamma$ for all implicit stages $j \geq 2$, enabling reuse of a single Helmholtz matrix across all stages. In addition, Fluca assumes that all IMEX schemes are of CK or ARS type.

### Stage Algorithm

First, let $d_j$ be defined by

```math
d_1 = 1, \quad d_j = -\frac{1}{\gamma} \sum_{k=1}^{j-1} a_{jk}d_k
```

**Stage 1** ($j = 1$, explicit, $a_{11} = 0$):

The first stage uses the solution from the previous time step directly:

```math
t_1 = t^{n-1}, \quad u_1 = u^{n-1}, \quad p_1 = p^{n-1}, \quad \hat{K}_1^u = C_1^u - Gp_1
```

For schemes of type CK that are not of type ARS, also evaluate:

```math
K_1^u = F_\text{diff}(u_1), \quad \tilde{\nu}_1 = D(K_1^u + \hat{K}_1^u) + \alpha D u_1
```

**Stages $j \geq 2$** (implicit):

Let $t_j = t^{n-1} + \tau c_j$ with $c_j = \sum_{k=1}^j a_{jk}$. Each subsequent stage proceeds through the following steps:

*Step 1 - Implicit velocity step*:

Accumulate contributions from previous stages:

```math
u_{j,*} = u^{n-1} + \tau \sum_{k=1}^{j-1} \left( a_{jk} K_k^u + \hat{a}_{jk} \hat{K}_k^u \right) \tag{7}
```

The explicit slopes $\hat{K}_k^u$ contain the pressure gradient $-Gp_k$, so pressure enters through the explicit accumulation.

Then solve:

```math
u_j = u_{j,*} + \hat{\tau} F_\text{diff}(u_j) \tag{8}
```

where $\hat{\tau} = a_{jj} \tau = \gamma \tau$.

*Step 2 - Pressure step*:

Define:

```math
\mu_j = (1 - a_{j1} \alpha \tau) p^{n-1} + \frac{1}{\gamma} \sum_{k=2}^{j-1} a_{jk}\tilde{\mu}_k \tag{9}
```

where $\tilde{\mu}_j = p_j - \mu_j$.

For schemes of type CK that are not of type ARS, let $\nu_j = d_j \tilde{\nu}_1$, and for schemes of type ARS, let $\nu_j = 0$.

Then let $\tilde{p}_j = \mu_j - \alpha \hat{\tau} p^{n-1}$ and solve:

```math
L(p_j - \tilde{p}_j) = D(K_j^u + C_j^u - G\tilde{p}_j) + \alpha D u^{n-1} - \nu_j \tag{10}
```

Note that the compact Laplacian $L$ on the LHS comes from letting $\sigma_0 = \hat{\tau}$. Otherwise it would have the form $DG - (\sigma_0/\hat{\tau})S$.

*Step 3 - Update velocity slopes*:

Evaluate:

```math
K_j^u = F_\text{diff}(u_j) = \frac{1}{\hat{\tau}} (u_j - u_{j,*}), \quad \hat{K}_j^u = C_j^u - Gp_j
```

### Solution Update

Put $p^n = p_s$, and evaluate:

```math
u^n = u^{n-1} + \tau \sum_{k=1}^s (b_k K_k^u + \hat{b}_k \hat{K}_k^u) \tag{11}
```

FSAL (first-same-as-last) schemes reuse the last explicit slope as the first slope of the next step.

### Stability

The SRK method has two stability constraints:

**CFL condition from the explicit pressure gradient.** The explicit treatment of $Gp$ introduces a CFL-like restriction $\tau \leq \text{CFL}_\text{max} / \lambda_\text{max}$, where $\lambda_\text{max}$ is the largest eigenvalue of the discrete pressure operator and $\text{CFL}_\text{max}$ depends on the scheme (see the table in the next section). In practice, this limits the time step relative to the grid spacing.

**Baumgarte parameter $\alpha$.** The spectral analysis of the SRK amplification matrix [1, Section 5.6] shows that stability requires

```math
0 \leq \alpha\tau \leq (\alpha\tau)_\text{max}
```

where $(\alpha\tau)_\text{max}$ depends on the IMEX scheme. For all schemes satisfying $b = \hat{b}$ (i.e., equal implicit and explicit weight vectors), the bound is $(\alpha\tau)_\text{max} = 2$. A few schemes that do not satisfy $b = \hat{b}$ have tighter limits: ARS(1,1,1) has $(\alpha\tau)_\text{max} = 1$, ARS(2,2,2) has $(\alpha\tau)_\text{max} \approx 0.82$, and ARS(4,4,3) has $(\alpha\tau)_\text{max} = 2$ for $r_\sigma = 0$ but $\approx 1.43$ for $r_\sigma = 1$. Following the numerical experiments in [1, Section 6], Fluca uses $\alpha\tau = 0.5(\alpha\tau)_\text{max}$.

Because $\sigma_0$ depends on the timestep $\tau$, changing $\tau$ between steps generates a non-physical perturbation to the stabilized continuity equation (4). All computations in [1] use a fixed timestep.

### Available Schemes

Fluca provides 12 IMEX RK schemes, selectable at runtime via `-seg_srk_type`:

| Name | Paper Name | Type | Order | Stages | $(\alpha\tau)_\text{max}$ | Source |
|------|-----------|------|-------|--------|------------------------|--------|
| `ars111` | ARS(1,1,1) | ARS | 1 | 2 | 1 | [2] |
| `ars121` | ARS(1,2,1) | ARS | 1 | 2 | 2 | [2] |
| `ars222` | ARS(2,2,2) | ARS | 2 | 3 | ~0.82 | [2] |
| `ars232` | ARS(2,3,2) | ARS | 2 | 3 | 2 | [2] |
| `ars343` | ARS(3,4,3) | ARS | 3 | 4 | 2 | [2] |
| `ars443` | ARS(4,4,3) | ARS | 3 | 5 | 2 | [2] |
| `mars343` | MARS(3,4,3) | ARS | 3 | 4 | 2 | [3] |
| `mark324l2sa` | MARK3(2)4L[2]SA | CK | 3 | 4 | 2 | [3] |
| `ark324l2sa` | ARK3(2)4L[2]SA | CK | 3 | 4 | 2 | [4] |
| `ark436l2sa` | ARK4(3)6L[2]SA | CK | 4 | 6 | 2 | [4] |
| `ark548l2sa` | ARK5(4)8L[2]SA | CK | 5 | 8 | 2 | [4] |
| `bhr553` | BHR(5,5,3) | CK | 3 | 5 | 2 | [5] |

The paper [1] recommends:

1. **ARS(3,4,3)** (`ars343`, default): Best ARS-type scheme; good behavior across all tests.
2. **ARK4(3)6L[2]SA** (`ark436l2sa`): Best overall; outperforms 3rd-order multistep methods.
3. **BHR(5,5,3)** (`bhr553`): No pressure order reduction; consider for time-dependent BCs.

### Properties

The SRK approach has several notable properties:

- **No saddle-point system**: Each stage solves only SPD systems (Helmholtz and Poisson), avoiding the indefinite coupled velocity-pressure system.
- **Constant SDIRK diagonal**: A single Helmholtz matrix is assembled and factored once per time step (or once total if $\tau$ is fixed), then reused for all stages.
- **High-order pressure**: Using the divergence of the full momentum residual as the pressure Poisson RHS (rather than $\text{div}(u)$) avoids the $O(1)$ pressure error of classical projection methods. However, most IMEX RK schemes suffer from *order reduction*, limiting pressure convergence to $O(\tau^2)$ regardless of the scheme's formal order. Among the schemes implemented in Fluca, only BHR(5,5,3) achieves full third-order pressure convergence [1].
- **CK-type efficiency**: For CK-type schemes, the SRK formulation calls the pressure solver only $s - 1$ times per timestep (the first stage is explicit and skips the pressure solve), compared to $s$ times for type A methods. This is a key computational advantage of the CK-based construction in [1] over the original SRK schemes of Colomés and Badia.
- **Explicit pressure gradient CFL**: The explicit treatment of $Gp$ introduces a CFL-like stability restriction. The maximum stable CFL number varies by scheme (see Table 1 of [1]).

## References

1. P. Bakhvalov, Segregated Runge-Kutta schemes for the time integration of the incompressible Navier-Stokes equations in presence of pressure stabilization, [arXiv:2506.09519](https://arxiv.org/abs/2506.09519) (2025).
2. U. M. Ascher, S. J. Ruuth, and R. J. Spiteri, Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations, *Appl. Numer. Math.*, 25, 151--167 (1997).
3. S. Boscarino and G. Russo, On the uniform accuracy of IMEX Runge-Kutta schemes and applications to hyperbolic systems with relaxation, Communications to SIMAI congress 2, 426, 11--33 (2007).
4. C. A. Kennedy and M. H. Carpenter, Additive Runge-Kutta schemes for convection-diffusion-reaction equations, *Appl. Numer. Math.*, 44, 139--181 (2003).
5. S. Boscarino, On an accurate third order implicit-explicit Runge-Kutta method for stiff problems, *Appl. Numer. Math.*, 59, 1515--1528 (2009).
