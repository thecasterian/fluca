#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test the FLUCAFDIBFORCING fallbacks at a domain wall\n"
                           "Options:\n"
                           "  -case <wall_dirichlet|zero_order|wall_apply> : fallback and check (default: wall_dirichlet)\n";

/* Solid for x > WALL_X0; the forcing cells at i = 0 have their fluid side off the grid */
#define WALL_X0 0.15625

static PetscErrorCode Wall_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = WALL_X0 - x[0];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal Clean_Private(PetscReal v)
{
  return PetscAbsReal(v) < 1e-12 ? 0. : v;
}

/* Component 0 = 1 - 3.2 x: 1 at the domain wall x = 0 and 0.5 at the immersed wall x = WALL_X0 */
static PetscErrorCode FillLinear_Private(DM dm, Vec u)
{
  Vec                 ul;
  PetscScalar      ***arr;
  const PetscScalar **cx, **cy;
  PetscInt            xs, ys, xm, ym, i, j, slot, cslot;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &cx, &cy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &cslot));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &slot));
  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMGetLocalVector(dm, &ul));
  PetscCall(VecZeroEntries(ul));
  PetscCall(DMStagVecGetArray(dm, ul, &arr));
  for (j = ys; j < ys + ym; ++j)
    for (i = xs; i < xs + xm; ++i) arr[j][i][slot] = 1. - 3.2 * PetscRealPart(cx[i][cslot]);
  PetscCall(DMStagVecRestoreArray(dm, ul, &arr));
  PetscCall(DMLocalToGlobal(dm, ul, INSERT_VALUES, u));
  PetscCall(DMRestoreLocalVector(dm, &ul));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &cx, &cy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintValue_Private(DM dm, Vec y, PetscInt i, PetscInt j, const char label[])
{
  DMStagStencil s;
  PetscScalar   v;
  Vec           yl;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(dm, &yl));
  PetscCall(DMGlobalToLocal(dm, y, INSERT_VALUES, yl));
  s.i   = i;
  s.j   = j;
  s.k   = 0;
  s.c   = 0;
  s.loc = DMSTAG_ELEMENT;
  PetscCall(DMStagVecGetValuesStencil(dm, yl, 1, &s, &v));
  PetscCall(DMRestoreLocalVector(dm, &yl));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s at (%" PetscInt_FMT ", %" PetscInt_FMT "): %.6f\n", label, i, j, (double)Clean_Private(PetscRealPart(v))));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintStencilAt_Private(FlucaFD fd, PetscInt i, PetscInt j)
{
  FlucaFDStencilPoint points[64];
  PetscInt            npoints;

  PetscFunctionBeginUser;
  PetscCall(FlucaFDGetStencil(fd, 0., i, j, 0, &npoints, points));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Stencil at (%" PetscInt_FMT ", %" PetscInt_FMT "):\n  npoints = %" PetscInt_FMT "\n", i, j, npoints));
  PetscCall(PrintStencil(2, npoints, points));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                       dm;
  Mesh                     mesh;
  IBBody                   body;
  FlucaFD                  fd;
  Vec                      u, y;
  char                     cs[64] = "wall_dirichlet";
  PetscBool                iszero, isapply;
  const PetscScalar        u_body[2] = {0.5, 0.};
  FlucaFDBoundaryCondition bcs[4]    = {{0}};

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-case", cs, sizeof(cs), NULL));
  PetscCall(PetscStrcmp(cs, "zero_order", &iszero));
  PetscCall(PetscStrcmp(cs, "wall_apply", &isapply));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 2, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &body));
  PetscCall(IBBodySetType(body, IBBODYLEVELSET));
  PetscCall(IBBodyLevelSetSetFunction(body, Wall_Private, NULL));
  PetscCall(IBBodySetVelocity(body, 2, u_body));
  PetscCall(MeshAddBody(mesh, body));
  PetscCall(MeshSetUp(mesh));

  PetscCall(FlucaFDIBForcingCreate(mesh, DMSTAG_ELEMENT, 0, 0, &fd));
  if (!iszero) {
    bcs[0].type  = FLUCAFD_BC_DIRICHLET; /* left wall, value 1 */
    bcs[0].value = 1.;
    PetscCall(FlucaFDSetBoundaryConditions(fd, 0, bcs));
  }
  PetscCall(FlucaFDSetFromOptions(fd));
  PetscCall(FlucaFDSetUp(fd));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &y));
  if (isapply) {
    PetscCall(FillLinear_Private(dm, u));
    PetscCall(VecZeroEntries(y));
    PetscCall(FlucaFDApply(fd, 0., dm, dm, u, y));
    PetscCall(PrintValue_Private(dm, y, 0, 4, "Residual, forcing at the wall"));
  } else PetscCall(PrintStencilAt_Private(fd, 0, 4));

  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&u));
  PetscCall(FlucaFDDestroy(&fd));
  PetscCall(IBBodyDestroy(&body));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: wall_dirichlet
    nsize: 1
    args: -case wall_dirichlet

  test:
    suffix: zero_order
    nsize: 1
    args: -case zero_order

  test:
    suffix: wall_apply
    nsize: 1
    args: -case wall_apply

TEST*/
