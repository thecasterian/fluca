#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test FLUCAFDIBFORCING: forcing and solid rows of the immersed boundary\n"
                           "Options:\n"
                           "  -case <wall|oblique|disk|prealloc> : geometry and check (default: wall)\n"
                           "  -flucafd_ib_interpolation_order <1|2>\n";

#define WALL_X0 0.28125

static PetscErrorCode Wall_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = x[0] - WALL_X0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Oblique_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = 0.6 * x[0] + 0.8 * x[1] - 0.7575;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Disk_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = PetscSqrtReal((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) - 0.3;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Rigid rotation about (0.5, 0.5) with angular velocity omega(t) = t */
static PetscErrorCode Rotation_Private(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar u[])
{
  PetscFunctionBeginUser;
  u[0] = -t * (x[1] - 0.5);
  u[1] = t * (x[0] - 0.5);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RotationDot_Private(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar u[])
{
  PetscFunctionBeginUser;
  u[0] = -(x[1] - 0.5);
  u[1] = x[0] - 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal Clean_Private(PetscReal v)
{
  return PetscAbsReal(v) < 1e-12 ? 0. : v;
}

/* Component 0 = (x - x0)^order + 0.5, or phi_oblique + 0.5 if oblique; component 1 = 0 */
static PetscErrorCode FillField_Private(DM dm, PetscInt order, PetscBool oblique, Vec u)
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
    for (i = xs; i < xs + xm; ++i) {
      if (oblique) arr[j][i][slot] = 0.6 * PetscRealPart(cx[i][cslot]) + 0.8 * PetscRealPart(cy[j][cslot]) - 0.7575 + 0.5;
      else arr[j][i][slot] = PetscPowRealInt(PetscRealPart(cx[i][cslot]) - WALL_X0, order) + 0.5;
    }
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
  DM                     dm;
  Mesh                   mesh;
  IBBody                 body;
  FlucaFD                fd;
  Vec                    u, y;
  Mat                    A;
  char                   cs[64] = "wall";
  PetscBool              isdisk, isprealloc, isoblique;
  PetscInt               N, order = 2, i, ir, gr, ncols, nnz;
  const PetscScalar     *vals;
  const PetscScalar      u_wall[2] = {0.5, 0.};
  MeshIBPointType        type;
  DMStagStencil          row;
  ISLocalToGlobalMapping ltog;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-case", cs, sizeof(cs), NULL));
  PetscCall(PetscStrcmp(cs, "disk", &isdisk));
  PetscCall(PetscStrcmp(cs, "prealloc", &isprealloc));
  PetscCall(PetscStrcmp(cs, "oblique", &isoblique));
  N = (isdisk || isoblique) ? 16 : 8; /* oblique: [0,2]^2 with h = 1/8, so that the wall leaves through no domain corner */

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, N, N, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 2, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., isoblique ? 2. : 1., 0., isoblique ? 2. : 1., 0., 0.));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &body));
  PetscCall(IBBodySetType(body, IBBODYLEVELSET));
  if (isdisk) {
    PetscCall(IBBodyLevelSetSetFunction(body, Disk_Private, NULL));
    PetscCall(IBBodySetVelocityFunction(body, Rotation_Private, RotationDot_Private, NULL));
  } else {
    PetscCall(IBBodyLevelSetSetFunction(body, isoblique ? Oblique_Private : Wall_Private, NULL));
    PetscCall(IBBodySetVelocity(body, 2, u_wall));
  }
  PetscCall(MeshAddBody(mesh, body));
  PetscCall(MeshSetUp(mesh));

  PetscCall(FlucaFDIBForcingCreate(mesh, DMSTAG_ELEMENT, 0, 0, &fd));
  PetscCall(FlucaFDSetFromOptions(fd));
  PetscCall(FlucaFDSetUp(fd));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-flucafd_ib_interpolation_order", &order, NULL));

  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &y));
  if (isprealloc) {
    PetscCall(DMCreateMatrix(dm, &A));
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
    PetscCall(FlucaFDGetOperator(fd, dm, dm, A));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    row.i   = 2;
    row.j   = 4;
    row.k   = 0;
    row.c   = 0;
    row.loc = DMSTAG_ELEMENT;
    PetscCall(DMStagStencilToIndexLocal(dm, 2, 1, &row, &ir));
    PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
    PetscCall(ISLocalToGlobalMappingApply(ltog, 1, &ir, &gr));
    /* The DMStag pattern stores explicit zeros, so count only the nonzero values */
    PetscCall(MatGetRow(A, gr, &ncols, NULL, &vals));
    for (i = 0, nnz = 0; i < ncols; ++i)
      if (vals[i] != 0.) ++nnz;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Forcing row (2, 4) fits the DMStag preallocation; nonzeros: %" PetscInt_FMT "\n", nnz));
    PetscCall(MatRestoreRow(A, gr, &ncols, NULL, &vals));
    PetscCall(MatDestroy(&A));
  } else if (isdisk) {
    PetscCall(VecZeroEntries(u));
    PetscCall(VecZeroEntries(y));
    PetscCall(FlucaFDApply(fd, 2., dm, dm, u, y));
    PetscCall(PrintValue_Private(dm, y, 8, 8, "Apply, solid"));
    for (i = 0; i < N; ++i) {
      PetscCall(MeshGetIBPointType(mesh, DMSTAG_ELEMENT, i, 8, 0, &type));
      if (type == MESH_IB_FORCING) break;
    }
    PetscCall(PrintValue_Private(dm, y, i, 8, "Apply, forcing"));
    PetscCall(VecZeroEntries(y));
    PetscCall(FlucaFDApplyDot(fd, 2., dm, dm, u, y));
    PetscCall(PrintValue_Private(dm, y, 8, 8, "ApplyDot, solid"));
    PetscCall(PrintValue_Private(dm, y, i, 8, "ApplyDot, forcing"));
  } else if (isoblique) {
    PetscCall(PrintStencilAt_Private(fd, 4, 4));
    PetscCall(FillField_Private(dm, order, PETSC_TRUE, u));
    PetscCall(VecZeroEntries(y));
    PetscCall(FlucaFDApply(fd, 0., dm, dm, u, y));
    PetscCall(PrintValue_Private(dm, y, 4, 4, "Residual, forcing"));
  } else {
    PetscCall(PrintStencilAt_Private(fd, 1, 4));
    PetscCall(PrintStencilAt_Private(fd, 2, 4));
    PetscCall(PrintStencilAt_Private(fd, 5, 4));
    PetscCall(FillField_Private(dm, order, PETSC_FALSE, u));
    PetscCall(VecZeroEntries(y));
    PetscCall(FlucaFDApply(fd, 0., dm, dm, u, y));
    PetscCall(PrintValue_Private(dm, y, 1, 4, "Residual, solid"));
    PetscCall(PrintValue_Private(dm, y, 2, 4, "Residual, forcing"));
    PetscCall(PrintValue_Private(dm, y, 5, 4, "Residual, fluid"));
  }

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
    suffix: wall_order_1
    nsize: 1
    args: -case wall -flucafd_ib_interpolation_order 1

  test:
    suffix: wall_order_2
    nsize: 1
    args: -case wall -flucafd_ib_interpolation_order 2

  test:
    suffix: oblique
    nsize: 1
    args: -case oblique

  test:
    suffix: disk
    nsize: 1
    args: -case disk

  test:
    suffix: prealloc
    nsize: 1
    args: -case prealloc

TEST*/
