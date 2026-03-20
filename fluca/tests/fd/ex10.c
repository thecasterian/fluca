#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test per-component boundary conditions in FlucaFDSum\n";

/*
  Input: 2 element DOFs on a 1D grid with DM_BOUNDARY_NONE.
    Component 0: u(x) = x^2
    Component 1: v(x) = sin(pi * x)

  fd_a = d/dx on component 0, fd_b = d/dx on component 1.
  fd_sum = fd_a + fd_b, with per-component BCs set on the sum:
    Component 0: Dirichlet u(0)=0, u(1)=1
    Component 1: Neumann   v'(0)=pi, v'(1)=-pi

  We print stencils at boundary-adjacent indices and verify that
  applying the sum matches applying each operand individually.
*/

static PetscErrorCode FillInputVector(DM dm, Vec u)
{
  Vec                 u_local;
  PetscInt            x, m, i, slot_elem0, slot_elem1;
  PetscScalar       **arr;
  const PetscScalar **arr_coord;
  PetscInt            slot_coord_elem;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &slot_coord_elem));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot_elem0));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 1, &slot_elem1));
  PetscCall(DMStagGetCorners(dm, &x, NULL, NULL, &m, NULL, NULL, NULL, NULL, NULL));

  PetscCall(DMGetLocalVector(dm, &u_local));
  PetscCall(VecZeroEntries(u_local));
  PetscCall(DMStagVecGetArray(dm, u_local, &arr));
  for (i = x; i < x + m; ++i) {
    PetscReal xi       = PetscRealPart(arr_coord[i][slot_coord_elem]);
    arr[i][slot_elem0] = xi * xi;
    arr[i][slot_elem1] = PetscSinReal(PETSC_PI * xi);
  }
  PetscCall(DMStagVecRestoreArray(dm, u_local, &arr));
  PetscCall(DMLocalToGlobal(dm, u_local, INSERT_VALUES, u));
  PetscCall(DMRestoreLocalVector(dm, &u_local));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                       dm_in, dm_out;
  FlucaFD                  fd_a, fd_b, fd_sum;
  FlucaFD                  operands[2];
  FlucaFDBoundaryCondition bcs_comp0[2] = {{0}}, bcs_comp1[2] = {{0}};
  Vec                      u, y_sum, y_a, y_b;
  PetscReal                norm_diff;
  FlucaFDStencilPoint      points[64];
  PetscInt                 npoints, M;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Input DM: 2 element DOFs (component 0 = u, component 1 = v) */
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 0, 2, DMSTAG_STENCIL_BOX, 1, NULL, &dm_in));
  PetscCall(DMSetUp(dm_in));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm_in, 0., 1., 0., 0., 0., 0.));

  /* Output DM: 1 element DOF */
  PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 0, 1, DMSTAG_STENCIL_BOX, 1, NULL, &dm_out));
  PetscCall(DMSetUp(dm_out));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm_out, 0., 1., 0., 0., 0., 0.));

  /* fd_a: d/dx on input component 0 -> output component 0 */
  PetscCall(FlucaFDDerivativeCreate(dm_in, FLUCAFD_X, 1, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &fd_a));
  /* fd_b: d/dx on input component 1 -> output component 0 */
  PetscCall(FlucaFDDerivativeCreate(dm_in, FLUCAFD_X, 1, 2, DMSTAG_ELEMENT, 1, DMSTAG_ELEMENT, 0, &fd_b));

  /* BCs for component 0 (u=x^2): Dirichlet u(0)=0, u(1)=1 */
  bcs_comp0[0].type  = FLUCAFD_BC_DIRICHLET;
  bcs_comp0[0].value = 0.;
  bcs_comp0[1].type  = FLUCAFD_BC_DIRICHLET;
  bcs_comp0[1].value = 1.;

  /* BCs for component 1 (v=sin(pi*x)): Neumann v'(0)=pi, v'(1)=-pi */
  bcs_comp1[0].type  = FLUCAFD_BC_NEUMANN;
  bcs_comp1[0].value = PETSC_PI;
  bcs_comp1[1].type  = FLUCAFD_BC_NEUMANN;
  bcs_comp1[1].value = -PETSC_PI;

  /* Set BCs on individual operators (for reference apply) */
  PetscCall(FlucaFDSetBoundaryConditions(fd_a, 0, bcs_comp0));
  PetscCall(FlucaFDSetBoundaryConditions(fd_b, 1, bcs_comp1));
  PetscCall(FlucaFDSetUp(fd_a));
  PetscCall(FlucaFDSetUp(fd_b));

  /* Create sum operator and set per-component BCs on root */
  operands[0] = fd_a;
  operands[1] = fd_b;
  PetscCall(FlucaFDSumCreate(2, operands, &fd_sum));
  PetscCall(FlucaFDSetBoundaryConditions(fd_sum, 0, bcs_comp0));
  PetscCall(FlucaFDSetBoundaryConditions(fd_sum, 1, bcs_comp1));
  PetscCall(FlucaFDSetUp(fd_sum));

  /* Print stencils at boundary-adjacent indices */
  PetscCall(DMStagGetGlobalSizes(dm_out, &M, NULL, NULL));

  PetscCall(FlucaFDGetStencil(fd_sum, 0., 0, 0, 0, &npoints, points));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Stencil at i=0 (left boundary):\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  npoints = %" PetscInt_FMT "\n", npoints));
  PetscCall(PrintStencil(1, npoints, points));

  PetscCall(FlucaFDGetStencil(fd_sum, 0., M - 1, 0, 0, &npoints, points));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Stencil at i=%" PetscInt_FMT " (right boundary):\n", M - 1));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  npoints = %" PetscInt_FMT "\n", npoints));
  PetscCall(PrintStencil(1, npoints, points));

  /* Apply sum operator */
  PetscCall(DMCreateGlobalVector(dm_in, &u));
  PetscCall(FillInputVector(dm_in, u));

  PetscCall(DMCreateGlobalVector(dm_out, &y_sum));
  PetscCall(FlucaFDApply(fd_sum, 0., dm_in, dm_out, u, y_sum));

  /* Apply individual operators and sum results manually */
  PetscCall(DMCreateGlobalVector(dm_out, &y_a));
  PetscCall(DMCreateGlobalVector(dm_out, &y_b));
  PetscCall(FlucaFDApply(fd_a, 0., dm_in, dm_out, u, y_a));
  PetscCall(FlucaFDApply(fd_b, 0., dm_in, dm_out, u, y_b));

  /* y_sum should equal y_a + y_b */
  PetscCall(VecAXPY(y_a, 1.0, y_b));
  PetscCall(VecAXPY(y_sum, -1.0, y_a));
  PetscCall(VecNorm(y_sum, NORM_INFINITY, &norm_diff));
  PetscCheck(norm_diff < 1e-10, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "||y_sum - (y_a + y_b)||_inf = %g exceeds tolerance", (double)norm_diff);

  PetscCall(VecDestroy(&y_b));
  PetscCall(VecDestroy(&y_a));
  PetscCall(VecDestroy(&y_sum));
  PetscCall(VecDestroy(&u));
  PetscCall(FlucaFDDestroy(&fd_sum));
  PetscCall(FlucaFDDestroy(&fd_b));
  PetscCall(FlucaFDDestroy(&fd_a));
  PetscCall(DMDestroy(&dm_out));
  PetscCall(DMDestroy(&dm_in));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: per_component_bcs
    nsize: 1
    args:

TEST*/
