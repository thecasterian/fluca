# Theory Guide

This document provides a comprehensive overview of the theoretical foundations and numerical methods implemented in Fluca. The guide is intended for users who wish to understand the mathematical and numerical principles underlying the software, as well as developers who are contributing to or extending the codebase.

## Table of Contents

- [Governing Equations](#governing-equations)
- [Grid System](#grid-system)
- [Temporal Discretization](#temporal-discretization)
- [Spatial Discretization](#spatial-discretization)
- [Immersed Boundary Method](#immersed-boundary-method)
- [Solvers](#solvers)

## Governing Equations

Fluca is designed to simulate unsteady incompressible viscous flow. Assuming constant density $\rho$ and constant dynamic viscosity $\mu$ (or equivalently, constant kinematic viscosity $\nu = \mu/\rho$), the governing equations consist of the continuity equation (mass conservation) and the Navier-Stokes equations (momentum conservation):

$$ \frac{\partial u_i}{\partial x_i} = 0 $$

$$ \frac{\partial u_i}{\partial t} + \frac{\partial}{\partial x_j} u_i u_j = -\frac{1}{\rho} \frac{\partial p}{\partial x_i} + \nu \frac{\partial}{\partial x_j} \frac{\partial}{\partial x_j} u_i $$

## Grid System

Fluca employs a non-staggered (collocated) grid system, where all variables such as velocity components and pressure are stored at the cell centers. This approach offers several advantages over staggered grids, including simplified implementation, easier handling of boundary conditions, and better suitability for unstructured or hybrid grids. The collocated arrangement also facilitates straightforward extension to three dimensions and complex geometries.

However, the traditional non-staggered grid is known to suffer from the checkerboard (odd-even decoupling) problem, where spurious pressure oscillations can appear in the solution. This instability arises because the pressure gradient at a cell center, calculated from neighboring cell center pressures, does not "see" the pressure at that very cell, allowing non-physical pressure modes to persist. To overcome this well-known issue, Fluca introduces a staggered face-normal velocity component, which provides the necessary coupling between pressure and velocity fields.

The face-normal velocity $U$ is defined as

$$ U = \mathbf{u}_\text{face} \cdot \mathbf{n} $$

where $\mathbf{u}_\text{face}$ is the velocity vector evaluated at the face center and $\mathbf{n}$ is the unit normal vector on the face. Since each face has two possible normal directions, a consistent choice must be made. On Cartesian grids, Fluca adopts the convention that the face normal vector coincides with the positive direction of the standard basis vector corresponding to the coordinate axis perpendicular to the face. This choice ensures consistency across all faces and simplifies implementation.

Critically, the face velocity is not simply linearly interpolated from the neighboring cell centers; instead, it is computed using the Rhie-Chow interpolation scheme, which adds a pressure-gradient correction term. This correction ensures proper coupling between the pressure and velocity fields, effectively suppressing the checkerboard instability while maintaining the advantages of the collocated grid. The details of this interpolation are discussed later in the temporal discretization section.

## Temporal Discretization

### Navier-Stokes Equation

Fluca employs a second-order accurate time advancement scheme based on the method introduced by Kim & Choi [1]. Achieving second-order temporal accuracy is crucial for minimizing numerical dissipation and dispersion errors, which is especially important for capturing unsteady flow phenomena such as vortex shedding and transition to turbulence. The scheme is designed to be unconditionally stable and suitable for both explicit and implicit treatments of different terms.

The discretization begins by applying the Crank-Nicolson scheme to both the convective and viscous terms of the Navier-Stokes equation. The Crank-Nicolson scheme is a centered, second-order accurate implicit method that averages the terms at time levels $n$ and $n+1$:

$$ \frac{u_i^{n+1} - u_i^n}{\Delta t} + \frac{1}{2} \frac{\partial}{\partial x_j} (u_i^{n+1} u_j^{n+1} + u_i^n u_j^n) = -\frac{1}{\rho} \frac{\partial p^{n+1/2}}{\partial x_i} + \frac{\nu}{2} \frac{\partial}{\partial x_j} \frac{\partial}{\partial x_j} (u_i^{n+1} + u_i^n) $$

Here, the pressure is evaluated at the _half_ time step $n+1/2$, which is consistent with the time-centered nature of the scheme. However, the product $u_i^{n+1} u_j^{n+1}$ in the convective term introduces a nonlinearity that would require iterative solution methods. To avoid this computational expense while maintaining second-order accuracy, the convective term is linearized using a Taylor series expansion:

$$ u_i^{n+1} u_j^{n+1} + u_i^n u_j^n = u_i^{n+1} u_j^n + u_i^n u_j^{n+1} + O(\Delta t^2) $$

This linearization is exact to second order in $\Delta t$, so the overall temporal accuracy of the scheme remains second-order. The linearized form allows the equations to be solved without inner iterations, significantly reducing computational cost. The resulting fully implicit time advancement scheme can be written as:

$$ \frac{u^{n+1} - u^n}{\Delta t} + \frac{1}{2} \frac{\partial}{\partial x_j} (u_i^{n+1} u_j^n + u_i^n u_j^{n+1}) = -\frac{1}{\rho} \frac{\partial p^{n+1/2}}{\partial x_i} + \frac{\nu}{2} \frac{\partial}{\partial x_j} \frac{\partial}{\partial x_j} (u_i^{n+1} + u_i^n) $$

Rearranging this equation to separate known quantities from unknowns, we obtain:

$$ u_i^{n+1} + \frac{\Delta t}{2} \frac{\partial}{\partial x_j} (u_i^{n+1} u_j^n + u_i^n u_j^{n+1}) - \frac{\nu \Delta t}{2} \frac{\partial}{\partial x_j} \frac{\partial}{\partial x_j} u_i^{n+1} + \frac{\Delta t}{\rho} \frac{\partial}{\partial x_i} (p^{n+1/2} - q) = u_i^n + \frac{\nu \Delta t}{2} \frac{\partial}{\partial x_j} \frac{\partial}{\partial x_j} u_i^n - \frac{\Delta t}{\rho} \frac{\partial q}{\partial x_i} $$

Here, $q$ is an auxiliary pressure-like scalar field used in the Rhie-Chow interpolation. The introduction of $q$ allows us to subtract and add specific pressure gradient terms to properly formulate the face velocity calculation. In Fluca's implementation, $q$ is set to the pressure at the previous half time step, $p^{n-1/2}$, except for the very first time step ($n=0$), when the initial pressure field $p^0$ is used instead, as it is the only available pressure field at that point.

The pressure field $p^{n+1}$ can be obtained by extrapolation from known pressure values at previous time steps, $p^{n+1/2}$ and $p^{n-1/2}$. Armfield & Street [3] showed that this extrapolation maintains second-order temporal accuracy for the pressure field.

### Rhie-Chow Interpolation

Based on the temporal discretization described above, the Rhie-Chow interpolation for computing the face velocity is given by:

$$ \mathbf{u}_\text{face}^{n+1} = \overline{\mathbf{u}}^{n+1} + \frac{\Delta t}{\rho} \left[ \overline{\nabla (p^{n+1/2} - q)} - \left. \nabla (p^{n+1/2} - q) \right|_\text{face} \right] $$

where $\overline{\phi}$ denotes the linear interpolation of the values of $\phi$ from the neighboring cell centers to the face. The first term $\overline{\mathbf{u}}^{n+1}$ is the straightforward linear interpolation of the cell-centered velocities. The second term is the pressure correction, which consists of the difference between the interpolated pressure gradient (from cell centers) and the pressure gradient directly calculated at the face. This correction term effectively couples the pressure field to the velocity field at the face, preventing the checkerboard oscillations.

Note that the pressure correction coefficient is fixed as $\Delta t / \rho$, unlike the classical Rhie-Chow interpolation which uses coefficients from the discretized momentum equation. This approach follows Zang et al. [2], who developed a similar formulation for collocated grids, though not explicitly presented as the form of the Rhie-Chow interpolation.

### Continuity Equation

The continuity equation is enforced at the new time step $n+1$:

$$ \frac{\partial u_i^{n+1}}{\partial x_i} = 0 $$

## Spatial Discretization

### Cartesian Grid

Fluca employs the finite difference method (FDM) for spatial discretization on Cartesian grids. The finite difference approach is well-suited for structured grids. The method discretizes spatial derivatives by approximating them with differences of function values at discrete grid points.

As an example, consider the second derivative of a variable $\phi$ in the x-direction at cell $(i, j)$ (assuming a 2D configuration for simplicity):

$$ \begin{aligned}
\left. \frac{\partial}{\partial x} \frac{\partial}{\partial x} \phi \right|_{i,j} & \approx \frac{\delta}{\delta x} \left( \frac{\delta}{\delta x} \phi_\text{cell} \right)_\text{face} \\
& = \frac{\left.\frac{\delta}{\delta x}\phi_\text{cell}\right|_{i+1/2,j} - \left.\frac{\delta}{\delta x}\phi_\text{cell}\right|_{i-1/2,j}}{\Delta x_{i-1/2,i+1/2}} \\
& = \frac{(\phi_{i+1,j} - \phi_{i,j}) / \Delta x_{i,i+1} - (\phi_{i,j} - \phi_{i-1,j}) / \Delta x_{i-1,i}}{\Delta x_{i-1/2,i+1/2}}
\end{aligned} $$

Here, $\delta$ denotes a finite difference operator, and the subscripts denote cell indices or face locations (indicated by half-integer indices). This nested difference approach &mdash; computing the derivative of a derivative &mdash; ensures consistency with the control volume formulation and guarantees second-order spatial accuracy, at least on uniform grids. For non-uniform grids, the scheme remains second-order provided the grid stretching is smooth. All other spatial derivatives appearing in the governing equations are discretized in a similar manner, maintaining second-order accuracy throughout.

When the value of a variable is needed at a face location, it is typically obtained through linear interpolation from the neighboring cell centers. However, for the convective and continuity terms, Fluca leverages the face-normal velocity $U$ computed via Rhie-Chow interpolation. This provides stronger coupling between the pressure and velocity fields [1]:

$$ \frac{\partial}{\partial x_j} (u_i^{n+1} u_j^n + u_i^n u_j^{n+1}) \approx \frac{\delta}{\delta x_j} (\overline{u}_i^{n+1} U^n + \overline{u}_i^n \overline{u}_j^{n+1}) $$

$$ \frac{\partial u_i^{n+1}}{\partial x_i} \approx \sum_i \frac{\delta U^{n+1}}{\delta x_i} $$

As a concrete example, the discrete divergence of velocity at cell $(i, j)$ in a 2D Cartesian grid is:

$$ \left( \frac{\delta U}{\delta x} + \frac{\delta U}{\delta y} \right)_{i,j} = \frac{U_{i+1/2,j} - U_{i-1/2,j}}{\Delta x_{i-1/2,i+1/2}} + \frac{U_{i,j+1/2} - U_{i,j-1/2}}{\Delta y_{j-1/2,j+1/2}} $$

## Immersed Boundary Method

<!-- TODO: -->

## Solvers

### Matrix Form of the Discretized Governing Equations

To facilitate the development and analysis of solution algorithms, it is convenient to express the discretized governing equations in matrix form using discrete operators. This abstract representation clarifies the structure of the coupled system and enables rigorous analysis of solution methods.

Let us define the following discrete operators. The operator $\mathbf{A}$ represents the implicit part of the momentum equation, including the identity (from the time derivative), the convective terms, and the viscous terms:

$$ (\mathbf{A}\mathbf{u}^{n+1})_i = u_i^{n+1} + \frac{\Delta t}{2} \frac{\delta}{\delta x_j} (\overline{u}_i^{n+1} U^n + \overline{u}_i^n \overline{u}_j^{n+1})  - \frac{\nu \Delta t}{2} \frac{\delta}{\delta x_j} \frac{\delta}{\delta x_j} u_i^{n+1}$$

The operator $\mathbf{G}$ represents the (scaled) cell-centered pressure gradient:

$$ (\mathbf{G}p)_i = \frac{\Delta t}{\rho} \frac{\delta p}{\delta x_i} $$

The operator $\mathbf{D}$ represents the discrete divergence operator applied to the face-normal velocities:

$$ \mathbf{D}U^{n+1} = \sum_i \frac{\delta U^{n+1}}{\delta x_i} $$

Using these operators, the discretized momentum equation can be written compactly as:

$$ \mathbf{A}\mathbf{u}^{n+1} + \mathbf{G}(p^{n+1/2} - q) = u_i^n + \frac{\nu \Delta t}{2} \frac{\delta}{\delta x_j} \frac{\delta}{\delta x_j} u_i^n - \frac{\Delta t}{\rho} \frac{\delta q}{\delta x_i} \equiv \mathbf{r} $$

where $\mathbf{r}$ represents the right-hand side consisting of known quantities from the previous time step. The discretized continuity equation becomes:

$$ \mathbf{D}U^{n+1} = 0 $$

To complete the operator notation, we express the Rhie-Chow interpolation using two additional operators. Let $\mathbf{T}$ be the operator that interpolates a cell-centered vector to the faces and takes its normal component:

$$ \mathbf{T}\mathbf{v} = \overline{\mathbf{v}} \cdot \mathbf{n} $$

and let $\mathbf{G}^\text{st}$ be the operator that computes the (scaled) face-normal pressure gradient directly at the face (the "staggered" gradient):

$$ \mathbf{G}^\text{st}p = \frac{\Delta t}{\rho} \left. \frac{\delta p}{\delta n} \right|_\text{face} $$

The Rhie-Chow interpolation formula can then be written as:

$$ U^{n+1} = \mathbf{T}\mathbf{u}^{n+1} + \mathbf{T}\mathbf{G}(p^{n+1/2} - q) - \mathbf{G}^\text{st}(p^{n+1/2} - q) $$

Combining these three equations (momentum, Rhie-Chow interpolation, and continuity), we obtain a fully coupled matrix system for the unknowns $\mathbf{u}^{n+1}$, $U^{n+1}$, and $p^{n+1/2} - q$:

$$ \begin{bmatrix} \mathbf{A} & 0 & \mathbf{G} \\ \mathbf{T} & -\mathbf{I} & \mathbf{T}\mathbf{G} - \mathbf{G}^\text{st} \\ 0 & \mathbf{D} & 0 \end{bmatrix} \begin{bmatrix} \mathbf{u}^{n+1} \\ U^{n+1} \\ p^{n+1/2} - q \end{bmatrix} = \begin{bmatrix} \mathbf{r} \\ 0 \\ 0 \end{bmatrix} $$

This is a large, sparse, saddle-point system. Solving it directly is computationally expensive, so Fluca employs various solution strategies based on approximate factorizations and iterative methods. Every solver type in Fluca ultimately solves this matrix system, either directly or through various preconditioners and approximate factorizations.

### Fractional Step Method

The fractional step method (FSM), also known as the projection method or pressure-correction method, is a widely used approach for solving the coupled velocity-pressure system in incompressible flow simulations. Rather than solving the fully coupled system directly, the FSM splits the solution process into sequential steps involving intermediate velocity fields. This decomposition dramatically reduces the computational cost compared to monolithic approaches while maintaining good accuracy.

Perot [4] provided important theoretical insight by showing that the FSM can be understood as an approximate block LU decomposition of the coupled matrix system. This perspective clarifies why the method works and how errors arise. For the matrix form derived above, Fluca uses the following block LU decomposition as a preconditioner:

$$ \mathbf{P} = \begin{bmatrix} \mathbf{A} & 0 & 0 \\ \mathbf{T} & -\mathbf{I} & 0 \\ 0 & \mathbf{D} & -\mathbf{D}\mathbf{G}^\text{st} \end{bmatrix} \begin{bmatrix} \mathbf{I} & 0 & \mathbf{G} \\ 0 & \mathbf{I} & \mathbf{G}^\text{st} \\ 0 & 0 & \mathbf{I} \end{bmatrix} = \begin{bmatrix} \mathbf{A} & 0 & \mathbf{A}\mathbf{G} \\ \mathbf{T} & -\mathbf{I} & \mathbf{T}\mathbf{G} - \mathbf{G}^\text{st} \\ 0 & \mathbf{D} & 0 \end{bmatrix} $$

This approximate factorization $\mathbf{P}$ is close to, but not exactly equal to, the original matrix $\mathbf{M}$. The difference between them is:

$$ \mathbf{P} - \mathbf{M} = \begin{bmatrix} 0 & 0 & (\mathbf{A} - \mathbf{I}) \mathbf{G} \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} $$

Examining this error term is crucial for understanding the accuracy of the method. By the definition of $\mathbf{A}$, the operator $(\mathbf{A} - \mathbf{I})$ contains terms proportional to $\Delta t$ (from the convective and viscous contributions). Additionally, the operator $\mathbf{G}$ itself is proportional to $\Delta t$. Therefore, the product $(\mathbf{A} - \mathbf{I})\mathbf{G}$ is $O(\Delta t^2)$, meaning that the approximation error introduced by using $\mathbf{P}$ instead of $\mathbf{M}$ is second-order in time. This ensures that the overall temporal accuracy of the solution remains second-order, consistent with the time discretization scheme. Consequently, $\mathbf{P}$ serves as an effective and efficient preconditioner for iteratively solving the full system, or it can be used directly as an approximate solver with controlled error.

## References

1. D. Kim and H. Choi, A Second-Order Time-Accurate Finite Volume Method for Unsteady Incompressible Flow on Hybrid Unstructured Grids, J. Comput. Phys., 162, 411&ndash;428 (2000).
2. Y. Zang, R. L. Street, and J. R. Koseff, A non-staggered grid, fractional step method for time-dependent incompressible Navier–Stokes equations in curvilinear coordinates, J. Comput. Phys., 114, 18&ndash;33 (1994).
3. S. Armfield and R. Street, The pressure accuracy of fractional-step methods for the Navier-Stokes equations on staggered grids, ANZIAM J., 44, C20&ndash;C39 (2003).
4. J. Perot, An analysis of the fractional step method, J. Comput. Phys., 108, 51&ndash;58 (1993).
