#include "segtest.h"

static const char help[] = "Test the momentum rows of the coupled system on a periodic 2D Taylor-Green field\n"
                           "With U = T u, the linearized convection satisfies J(u) u = 2 div(ubar ubar),\n"
                           "so the momentum rows of M X and f have closed forms on a uniform grid.\n";

static PetscScalar TGV(PetscInt d, PetscReal x, PetscReal y)
{
  return d == 0 ? -PetscCosReal(x) * PetscSinReal(y) : PetscSinReal(x) * PetscCosReal(y);
}

int main(int argc, char **argv)
{
  DM                dm, sol_dm;
  Phys              phys;
  Seg               seg;
  Mat               M;
  Vec               X, f, MX, E_MX, E_f;
  PhysFieldLocation loc;
  PetscInt          c_vel, c_U, N, xs, ys, xm, ym, i, j, d;
  PetscReal         rho = 2., mu = 0.3, dt = 0.1, nu, h;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 4, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 2. * PETSC_PI, 0., 2. * PETSC_PI, 0., 0.));
  PetscCall(SegTestSetUp(dm, rho, mu, NULL, &phys, &seg));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, &loc, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, &loc, &c_U, NULL));
  PetscCall(DMStagGetGlobalSizes(sol_dm, &N, NULL, NULL));
  h  = 2. * PETSC_PI / N;
  nu = mu / rho;

  PetscCall(SegTestCreateSystem(phys, &M, &f));
  PetscCall(DMCreateGlobalVector(sol_dm, &X));
  PetscCall(DMCreateGlobalVector(sol_dm, &MX));
  PetscCall(DMCreateGlobalVector(sol_dm, &E_MX));
  PetscCall(DMCreateGlobalVector(sol_dm, &E_f));
  PetscCall(VecZeroEntries(X));
  PetscCall(VecZeroEntries(E_MX));
  PetscCall(VecZeroEntries(E_f));

  /* X: u = TGV at cell centers, U = linear interpolation of the normal component, q = 0.
     E_MX = u + (dt/2) * 2 div(ubar ubar) - (dt/2) nu lap(u), E_f = u + (dt/2) nu lap(u). */
  PetscCall(DMStagGetCorners(sol_dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  for (j = ys; j < ys + ym; ++j) {
    for (i = xs; i < xs + xm; ++i) {
      PetscReal     xc = (i + 0.5) * h, yc = (j + 0.5) * h;
      DMStagStencil st;
      PetscScalar   v;

      st.i = i;
      st.j = j;
      st.k = 0;
      for (d = 0; d < 2; ++d) {
        PetscScalar u = TGV(d, xc, yc), lap, flux_p, flux_m, conv = 0.;

        st.loc = DMSTAG_ELEMENT;
        st.c   = c_vel + d;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &u, INSERT_VALUES));
        lap    = (TGV(d, xc + h, yc) + TGV(d, xc - h, yc) + TGV(d, xc, yc + h) + TGV(d, xc, yc - h) - 4. * u) / (h * h);
        flux_p = 0.25 * (u + TGV(d, xc + h, yc)) * (TGV(0, xc, yc) + TGV(0, xc + h, yc));
        flux_m = 0.25 * (u + TGV(d, xc - h, yc)) * (TGV(0, xc, yc) + TGV(0, xc - h, yc));
        conv += (flux_p - flux_m) / h;
        flux_p = 0.25 * (u + TGV(d, xc, yc + h)) * (TGV(1, xc, yc) + TGV(1, xc, yc + h));
        flux_m = 0.25 * (u + TGV(d, xc, yc - h)) * (TGV(1, xc, yc) + TGV(1, xc, yc - h));
        conv += (flux_p - flux_m) / h;
        v = u + dt * conv - 0.5 * dt * nu * lap;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, E_MX, 1, &st, &v, INSERT_VALUES));
        v = u + 0.5 * dt * nu * lap;
        PetscCall(DMStagVecSetValuesStencil(sol_dm, E_f, 1, &st, &v, INSERT_VALUES));
      }
      st.loc = DMSTAG_LEFT;
      st.c   = c_U;
      v      = 0.5 * (TGV(0, xc - h, yc) + TGV(0, xc, yc));
      PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
      st.loc = DMSTAG_DOWN;
      v      = 0.5 * (TGV(1, xc, yc - h) + TGV(1, xc, yc));
      PetscCall(DMStagVecSetValuesStencil(sol_dm, X, 1, &st, &v, INSERT_VALUES));
    }
  }
  PetscCall(VecAssemblyBegin(X));
  PetscCall(VecAssemblyEnd(X));
  PetscCall(VecAssemblyBegin(E_MX));
  PetscCall(VecAssemblyEnd(E_MX));
  PetscCall(VecAssemblyBegin(E_f));
  PetscCall(VecAssemblyEnd(E_f));

  PetscCall(SegCNLinearComputeMomentumSystem_Internal(seg, 0., dt, X, M, f));
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatMult(M, X, MX));

  PetscCall(SegTestCheckField(phys, PHYS_FIELD_VELOCITY, MX, E_MX, 1e-12));
  PetscCall(SegTestCheckField(phys, PHYS_FIELD_VELOCITY, f, E_f, 1e-12));
  PetscCall(SegTestCheckField(phys, PHYS_FIELD_PRESSURE, f, E_f, 0.));
  PetscCall(SegTestCheckField(phys, PHYS_FIELD_FACE_VELOCITY, f, E_f, 0.));

  PetscCall(VecDestroy(&E_f));
  PetscCall(VecDestroy(&E_MX));
  PetscCall(VecDestroy(&MX));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&M));
  PetscCall(SegDestroy(&seg));
  PetscCall(PhysDestroy(&phys));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: tgv
    nsize: 1
    output_file: output/empty.out

TEST*/
