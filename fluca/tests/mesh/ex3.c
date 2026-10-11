#include <flucamesh.h>
#include <flucasys.h>
#include <fluca/private/meshimpl.h>
#include <petscdmstag.h>

static const char help[] = "Test Mesh immersed-boundary classification from level-set bodies\n"
                           "Options:\n"
                           "  -case <circle2d|sphere3d|twocircles|wall|none|noface> : geometry (default: circle2d;\n"
                           "    noface is circle2d on a DM without face dofs)\n"
                           "  (DMStag options such as -stag_grid_x set the grid)\n";

static PetscErrorCode Circle_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  const PetscReal *c = (const PetscReal *)ctx; /* centre x, centre y, radius */

  PetscFunctionBeginUser;
  *phi = PetscSqrtReal((x[0] - c[0]) * (x[0] - c[0]) + (x[1] - c[1]) * (x[1] - c[1])) - c[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Sphere_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = PetscSqrtReal((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5) + (x[2] - 0.5) * (x[2] - 0.5)) - 0.3;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Solid for x < 0.28125: element centres of an 8-cell grid on [0, 1] give theta = 0.25 */
static PetscErrorCode Wall_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = x[0] - 0.28125;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal Clean_Private(PetscReal v)
{
  return PetscAbsReal(v) < 1e-12 ? 0. : v;
}

static PetscErrorCode AddLevelSetBody_Private(Mesh mesh, IBBodyLevelSetFn *fn, void *ctx)
{
  IBBody body;

  PetscFunctionBeginUser;
  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &body));
  PetscCall(IBBodySetType(body, IBBODYLEVELSET));
  PetscCall(IBBodyLevelSetSetFunction(body, fn, ctx));
  PetscCall(MeshAddBody(mesh, body));
  PetscCall(IBBodyDestroy(&body));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Owned index range of a location (faces normal to d include the last face on the last rank) */
static PetscErrorCode GetOwnedRange_Private(DM dm, DMStagStencilLocation loc, PetscInt s[], PetscInt e[])
{
  PetscInt dim, m[3], ne[3], d;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetCorners(dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2], &ne[0], &ne[1], &ne[2]));
  for (d = 0; d < 3; ++d) {
    if (d >= dim) {
      s[d] = 0;
      e[d] = 1;
    } else e[d] = s[d] + m[d] + (MeshIBIsFaceInDir_Private(loc, d) ? ne[d] : 0);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintCounts_Private(Mesh mesh, DMStagStencilLocation loc)
{
  DM              dm;
  PetscInt        s[3], e[3], i, j, k, cnt[3] = {0, 0, 0}, sum[3];
  MeshIBPointType type;

  PetscFunctionBeginUser;
  PetscCall(MeshGetDM(mesh, &dm));
  PetscCall(GetOwnedRange_Private(dm, loc, s, e));
  for (k = s[2]; k < e[2]; ++k)
    for (j = s[1]; j < e[1]; ++j)
      for (i = s[0]; i < e[0]; ++i) {
        PetscCall(MeshGetIBPointType(mesh, loc, i, j, k, &type));
        ++cnt[type];
      }
  PetscCallMPI(MPIU_Allreduce(cnt, sum, 3, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: fluid %" PetscInt_FMT ", forcing %" PetscInt_FMT ", solid %" PetscInt_FMT "\n", DMStagStencilLocations[loc], sum[0], sum[1], sum[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 2D, one rank: '.' fluid, 'f' forcing, '#' solid, rows from the top; or the body id of every point */
static PetscErrorCode PrintMap_Private(Mesh mesh, DMStagStencilLocation loc, PetscBool body_ids)
{
  DM              dm;
  PetscInt        s[3], e[3], i, j, body;
  MeshIBPointType type;
  const char      sym[3] = {'.', 'f', '#'};

  PetscFunctionBeginUser;
  PetscCall(MeshGetDM(mesh, &dm));
  PetscCall(GetOwnedRange_Private(dm, loc, s, e));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s %s:\n", DMStagStencilLocations[loc], body_ids ? "body ids" : "map"));
  for (j = e[1] - 1; j >= s[1]; --j) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  "));
    for (i = s[0]; i < e[0]; ++i) {
      if (body_ids) {
        PetscCall(MeshIBGetBodyId_Internal(mesh, loc, i, j, 0, &body));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%" PetscInt_FMT, body));
      } else {
        PetscCall(MeshGetIBPointType(mesh, loc, i, j, 0, &type));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%c", sym[type]));
      }
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* One rank: the first max_print forcing records of a location, in index order */
static PetscErrorCode PrintRecords_Private(Mesh mesh, DMStagStencilLocation loc, PetscInt max_print)
{
  DM                        dm;
  PetscInt                  dim, s[3], e[3], i, j, k, n, d, nprint = 0;
  const MeshIBForcingPoint *fp;

  PetscFunctionBeginUser;
  PetscCall(MeshGetDM(mesh, &dm));
  PetscCall(MeshGetDimension(mesh, &dim));
  PetscCall(GetOwnedRange_Private(dm, loc, s, e));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s forcing records:\n", DMStagStencilLocations[loc]));
  for (k = s[2]; k < e[2]; ++k)
    for (j = s[1]; j < e[1]; ++j)
      for (i = s[0]; i < e[0]; ++i) {
        if (nprint >= max_print) continue;
        PetscCall(MeshIBGetForcingPoint_Internal(mesh, loc, i, j, k, &fp));
        if (!fp) continue;
        ++nprint;
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "): normal (", i, j, k));
        for (d = 0; d < dim; ++d) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%.6f", d ? ", " : "", (double)Clean_Private(fp->normal[d])));
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, ")\n"));
        for (n = 0; n < fp->ncut; ++n) {
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, "    cut dir %" PetscInt_FMT " side %" PetscInt_FMT " theta %.6f body %" PetscInt_FMT " x_ib (", fp->cuts[n].dir, fp->cuts[n].side, (double)fp->cuts[n].theta, fp->cuts[n].body));
          for (d = 0; d < dim; ++d) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%.6f", d ? ", " : "", (double)Clean_Private(fp->cuts[n].x_ib[d])));
          PetscCall(PetscPrintf(PETSC_COMM_WORLD, ")\n"));
        }
      }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  Mesh        mesh;
  Vec         phi;
  char        cs[64]    = "circle2d";
  PetscReal   circle[3] = {0.5, 0.5, 0.3}, left[3] = {0.3, 0.5, 0.15}, right[3] = {0.7, 0.5, 0.15};
  PetscBool   is3d, istwo, iswall, isnone, isnoface;
  PetscMPIInt size;
  PetscInt    nb;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-case", cs, sizeof(cs), NULL));
  PetscCall(PetscStrcmp(cs, "sphere3d", &is3d));
  PetscCall(PetscStrcmp(cs, "twocircles", &istwo));
  PetscCall(PetscStrcmp(cs, "wall", &iswall));
  PetscCall(PetscStrcmp(cs, "none", &isnone));
  PetscCall(PetscStrcmp(cs, "noface", &isnoface));

  if (is3d) PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, 8, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, NULL, &dm));
  else PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, iswall ? 8 : 12, iswall ? 8 : 12, PETSC_DECIDE, PETSC_DECIDE, 0, isnoface ? 0 : 1, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 1.));
  PetscCall(MeshCartesianCreate(dm, &mesh));

  if (is3d) PetscCall(AddLevelSetBody_Private(mesh, Sphere_Private, NULL));
  else if (istwo) {
    PetscCall(AddLevelSetBody_Private(mesh, Circle_Private, left));
    PetscCall(AddLevelSetBody_Private(mesh, Circle_Private, right));
  } else if (iswall) PetscCall(AddLevelSetBody_Private(mesh, Wall_Private, NULL));
  else if (!isnone) PetscCall(AddLevelSetBody_Private(mesh, Circle_Private, circle));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));

  PetscCall(MeshGetNumBodies(mesh, &nb));
  PetscCall(MeshGetIBLevelSet(mesh, &phi));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "bodies: %" PetscInt_FMT ", level set: %s\n", nb, phi ? "yes" : "none"));
  PetscCall(PrintCounts_Private(mesh, DMSTAG_ELEMENT));
  PetscCall(PrintCounts_Private(mesh, DMSTAG_LEFT));
  PetscCall(PrintCounts_Private(mesh, DMSTAG_DOWN));
  if (is3d) PetscCall(PrintCounts_Private(mesh, DMSTAG_BACK));
  if (size == 1 && !is3d && !isnone && !isnoface) {
    PetscCall(PrintMap_Private(mesh, DMSTAG_ELEMENT, PETSC_FALSE));
    PetscCall(PrintMap_Private(mesh, DMSTAG_LEFT, PETSC_FALSE));
    PetscCall(PrintMap_Private(mesh, DMSTAG_DOWN, PETSC_FALSE));
  }
  if (size == 1 && istwo) PetscCall(PrintMap_Private(mesh, DMSTAG_ELEMENT, PETSC_TRUE));
  if (size == 1 && is3d) PetscCall(PrintRecords_Private(mesh, DMSTAG_ELEMENT, 3));
  if (size == 1 && iswall) {
    PetscCall(PrintRecords_Private(mesh, DMSTAG_ELEMENT, 2));
    PetscCall(PrintRecords_Private(mesh, DMSTAG_LEFT, 2));
  }

  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: circle2d
    nsize: 1
    args: -case circle2d

  test:
    suffix: sphere3d
    nsize: 1
    args: -case sphere3d

  test:
    suffix: twocircles
    nsize: 1
    args: -case twocircles

  test:
    suffix: wall
    nsize: 1
    args: -case wall

  test:
    suffix: none
    nsize: 1
    args: -case none

  test:
    suffix: noface
    nsize: 1
    args: -case noface

TEST*/
