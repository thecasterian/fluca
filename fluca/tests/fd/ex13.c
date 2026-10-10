#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test FLUCAFDIBMASK: an operand's rows at fluid points only, summed with FLUCAFDIBFORCING\n";

/* Solid for x > 0.71875: element 6 of 8 is solid, element 5 is a forcing point with theta = 0.25 */
static PetscErrorCode Wall_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = 0.71875 - x[0];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintStencilAt_Private(FlucaFD fd, const char label[], PetscInt i, PetscInt j)
{
  FlucaFDStencilPoint points[64];
  PetscInt            npoints;

  PetscFunctionBeginUser;
  PetscCall(FlucaFDGetStencil(fd, 0., i, j, 0, &npoints, points));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s stencil at (%" PetscInt_FMT ", %" PetscInt_FMT "):\n  npoints = %" PetscInt_FMT "\n", label, i, j, npoints));
  PetscCall(PrintStencil(2, npoints, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                       dm;
  Mesh                     mesh;
  IBBody                   body;
  FlucaFD                  op, mask, forcing, sum, terms[2];
  FlucaFDBoundaryCondition bcs[4] = {{0}};
  FlucaFDStencilPoint      points[64];
  PetscInt                 i, nop, nmask, nsum;
  MeshIBPointType          type;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &body));
  PetscCall(IBBodySetType(body, IBBODYLEVELSET));
  PetscCall(IBBodyLevelSetSetFunction(body, Wall_Private, NULL));
  PetscCall(MeshAddBody(mesh, body));
  PetscCall(MeshSetUp(mesh));

  /* d2/dx2 with a Dirichlet left wall, so row 0 carries a domain boundary point */
  PetscCall(FlucaFDDerivativeCreate(mesh, FLUCAFD_X, 2, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &op));
  bcs[0].type  = FLUCAFD_BC_DIRICHLET;
  bcs[0].value = 1.;
  PetscCall(FlucaFDSetBoundaryConditions(op, 0, bcs));
  PetscCall(FlucaFDSetUp(op));
  PetscCall(FlucaFDIBMaskCreate(op, &mask));
  PetscCall(FlucaFDSetUp(mask));
  PetscCall(FlucaFDIBForcingCreate(mesh, DMSTAG_ELEMENT, 0, 0, &forcing));
  PetscCall(FlucaFDSetUp(forcing));
  terms[0] = mask;
  terms[1] = forcing;
  PetscCall(FlucaFDSumCreate(2, terms, &sum));
  PetscCall(FlucaFDSetBoundaryConditions(sum, 0, bcs)); /* the root operator's BCs resolve off-grid points */
  PetscCall(FlucaFDSetUp(sum));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Row sizes along j = 4:\n"));
  for (i = 0; i < 8; ++i) {
    PetscCall(MeshGetIBPointType(mesh, DMSTAG_ELEMENT, i, 4, 0, &type));
    PetscCall(FlucaFDGetStencil(op, 0., i, 4, 0, &nop, points));
    PetscCall(FlucaFDGetStencil(mask, 0., i, 4, 0, &nmask, points));
    PetscCall(FlucaFDGetStencil(sum, 0., i, 4, 0, &nsum, points));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  i = %" PetscInt_FMT " %-7s operand %" PetscInt_FMT ", mask %" PetscInt_FMT ", mask + forcing %" PetscInt_FMT "\n", i, MeshIBPointTypes[type], nop, nmask, nsum));
  }
  PetscCall(PrintStencilAt_Private(op, "Operand", 0, 4));
  PetscCall(PrintStencilAt_Private(mask, "Mask", 0, 4));
  PetscCall(PrintStencilAt_Private(sum, "Mask + forcing", 4, 4));
  PetscCall(PrintStencilAt_Private(sum, "Mask + forcing", 5, 4));

  PetscCall(FlucaFDDestroy(&sum));
  PetscCall(FlucaFDDestroy(&forcing));
  PetscCall(FlucaFDDestroy(&mask));
  PetscCall(FlucaFDDestroy(&op));
  PetscCall(IBBodyDestroy(&body));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: wall
    nsize: 1
    args:

TEST*/
