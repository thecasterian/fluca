#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"

static const char help[] = "Test FLUCAFDIBSELECT: one operand per IB point type, each row with its operand's boundary conditions\n"
                           "Options:\n"
                           "  -test_case <wall|bc_per_type|nested> : case to run\n";

/* Solid for x > 0.71875: element 6 of 8 is solid, element 5 is a forcing point with theta = 0.25 */
static PetscErrorCode Wall_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = 0.71875 - x[0];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solid slab around x = 0.1875: element 1 is solid, elements 0 and 2 are forcing points */
static PetscErrorCode Slab_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = PetscAbsReal(x[0] - 0.1875) - 0.03125;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh_Private(IBBodyLevelSetFn *fn, DM *dm, IBBody *body, Mesh *mesh)
{
  PetscFunctionBeginUser;
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, dm));
  PetscCall(DMSetUp(*dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(*dm, 0., 1., 0., 1., 0., 0.));
  PetscCall(MeshCartesianCreate(*dm, mesh));
  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, body));
  PetscCall(IBBodySetType(*body, IBBODYLEVELSET));
  PetscCall(IBBodyLevelSetSetFunction(*body, fn, NULL));
  PetscCall(MeshAddBody(*mesh, *body));
  PetscCall(MeshSetUp(*mesh));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* d2/dx2 on elements with Dirichlet values on the left and right walls */
static PetscErrorCode CreateD2_Private(Mesh mesh, PetscScalar left, PetscScalar right, FlucaFD *fd)
{
  FlucaFDBoundaryCondition bcs[4] = {{0}};

  PetscFunctionBeginUser;
  PetscCall(FlucaFDDerivativeCreate(mesh, FLUCAFD_X, 2, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, fd));
  bcs[0].type  = FLUCAFD_BC_DIRICHLET;
  bcs[0].value = left;
  bcs[1].type  = FLUCAFD_BC_DIRICHLET;
  bcs[1].value = right;
  PetscCall(FlucaFDSetBoundaryConditions(*fd, 0, bcs));
  PetscCall(FlucaFDSetUp(*fd));
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

/* Select(op, NULL, NULL) keeps op's rows at fluid points only; Select(op, forcing, forcing) adds the forcing rows */
static PetscErrorCode TestWall_Private(void)
{
  DM                  dm;
  Mesh                mesh;
  IBBody              body;
  FlucaFD             op, forcing, masked, sel;
  FlucaFDStencilPoint points[64];
  PetscInt            i, nop, nmasked, nsel;
  MeshIBPointType     type;

  PetscFunctionBeginUser;
  PetscCall(CreateMesh_Private(Wall_Private, &dm, &body, &mesh));
  PetscCall(FlucaFDDerivativeCreate(mesh, FLUCAFD_X, 2, 2, DMSTAG_ELEMENT, 0, DMSTAG_ELEMENT, 0, &op));
  {
    FlucaFDBoundaryCondition bcs[4] = {{0}};

    /* Dirichlet left wall, so row 0 carries a domain boundary point */
    bcs[0].type  = FLUCAFD_BC_DIRICHLET;
    bcs[0].value = 1.;
    PetscCall(FlucaFDSetBoundaryConditions(op, 0, bcs));
  }
  PetscCall(FlucaFDSetUp(op));
  PetscCall(FlucaFDIBForcingCreate(mesh, DMSTAG_ELEMENT, 0, 0, &forcing));
  PetscCall(FlucaFDSetUp(forcing));
  PetscCall(FlucaFDIBSelectCreate(op, NULL, NULL, &masked));
  PetscCall(FlucaFDSetUp(masked));
  PetscCall(FlucaFDIBSelectCreate(op, forcing, forcing, &sel));
  PetscCall(FlucaFDSetUp(sel));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Row sizes along j = 4:\n"));
  for (i = 0; i < 8; ++i) {
    PetscCall(MeshGetIBPointType(mesh, DMSTAG_ELEMENT, i, 4, 0, &type));
    PetscCall(FlucaFDGetStencil(op, 0., i, 4, 0, &nop, points));
    PetscCall(FlucaFDGetStencil(masked, 0., i, 4, 0, &nmasked, points));
    PetscCall(FlucaFDGetStencil(sel, 0., i, 4, 0, &nsel, points));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  i = %" PetscInt_FMT " %-7s operand %" PetscInt_FMT ", fluid only %" PetscInt_FMT ", select %" PetscInt_FMT "\n", i, MeshIBPointTypes[type], nop, nmasked, nsel));
  }
  PetscCall(PrintStencilAt_Private(op, "Operand", 0, 4));
  PetscCall(PrintStencilAt_Private(masked, "Fluid only", 0, 4));
  PetscCall(PrintStencilAt_Private(sel, "Select", 4, 4));
  PetscCall(PrintStencilAt_Private(sel, "Select", 5, 4));

  PetscCall(FlucaFDDestroy(&sel));
  PetscCall(FlucaFDDestroy(&masked));
  PetscCall(FlucaFDDestroy(&forcing));
  PetscCall(FlucaFDDestroy(&op));
  PetscCall(IBBodyDestroy(&body));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* FLUID and FORCING operands differ only in their wall values; applied to zero, each wall row yields its own operand's value */
static PetscErrorCode TestBCPerType_Private(void)
{
  DM              dm;
  Mesh            mesh;
  IBBody          body;
  FlucaFD         fluid, forcing, sel;
  Vec             x, y, y_local;
  DMStagStencil   pos;
  PetscScalar     val;
  PetscInt        i;
  MeshIBPointType type;

  PetscFunctionBeginUser;
  PetscCall(CreateMesh_Private(Slab_Private, &dm, &body, &mesh));
  PetscCall(CreateD2_Private(mesh, 1., 3., &fluid));
  PetscCall(CreateD2_Private(mesh, 2., 4., &forcing));
  PetscCall(FlucaFDIBSelectCreate(fluid, forcing, NULL, &sel));
  PetscCall(FlucaFDSetUp(sel));

  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecZeroEntries(x));
  PetscCall(FlucaFDApply(sel, 0., dm, dm, x, y));
  PetscCall(DMGetLocalVector(dm, &y_local));
  PetscCall(DMGlobalToLocal(dm, y, INSERT_VALUES, y_local));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Select applied to zero along j = 4 (fluid walls 1, 3; forcing walls 2, 4):\n"));
  pos.loc = DMSTAG_ELEMENT;
  pos.j   = 4;
  pos.k   = 0;
  pos.c   = 0;
  for (i = 0; i < 8; ++i) {
    pos.i = i;
    PetscCall(MeshGetIBPointType(mesh, DMSTAG_ELEMENT, i, 4, 0, &type));
    PetscCall(DMStagVecGetValuesStencil(dm, y_local, 1, &pos, &val));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  i = %" PetscInt_FMT " %-7s %g\n", i, MeshIBPointTypes[type], (double)PetscRealPart(val)));
  }
  PetscCall(DMRestoreLocalVector(dm, &y_local));

  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(FlucaFDDestroy(&sel));
  PetscCall(FlucaFDDestroy(&forcing));
  PetscCall(FlucaFDDestroy(&fluid));
  PetscCall(IBBodyDestroy(&body));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A select must be the outermost operator: setting up a sum over it fails */
static PetscErrorCode TestNested_Private(void)
{
  DM             dm;
  Mesh           mesh;
  IBBody         body;
  FlucaFD        op, sel, sum;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCall(CreateMesh_Private(Wall_Private, &dm, &body, &mesh));
  PetscCall(CreateD2_Private(mesh, 0., 0., &op));
  PetscCall(FlucaFDIBSelectCreate(op, NULL, NULL, &sel));
  PetscCall(FlucaFDSetUp(sel));
  PetscCall(FlucaFDSumCreate(1, &sel, &sum));
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  ierr = FlucaFDSetUp(sum); /* expected to fail: the error code is the result under test */
  PetscCall(PetscPopErrorHandler());
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Sum over a select: %s\n", ierr ? "rejected" : "accepted"));

  PetscCall(FlucaFDDestroy(&sum));
  PetscCall(FlucaFDDestroy(&sel));
  PetscCall(FlucaFDDestroy(&op));
  PetscCall(IBBodyDestroy(&body));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  char      test_case[64] = "wall";
  PetscBool iswall, isbc, isnested;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-test_case", test_case, sizeof(test_case), NULL));
  PetscCall(PetscStrcmp(test_case, "wall", &iswall));
  PetscCall(PetscStrcmp(test_case, "bc_per_type", &isbc));
  PetscCall(PetscStrcmp(test_case, "nested", &isnested));
  if (iswall) PetscCall(TestWall_Private());
  else if (isbc) PetscCall(TestBCPerType_Private());
  else if (isnested) PetscCall(TestNested_Private());
  else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Unknown -test_case %s", test_case);
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: wall
    nsize: 1
    args: -test_case wall

  test:
    suffix: bc_per_type
    nsize: 1
    args: -test_case bc_per_type

  test:
    suffix: nested
    nsize: 1
    args: -test_case nested

TEST*/
