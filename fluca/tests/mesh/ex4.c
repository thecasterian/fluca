#include <flucamesh.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <fluca/private/meshimpl.h>

static const char help[] = "Test the surface IBBody: classification against an equivalent level set\n"
                           "Options:\n"
                           "  -case <polyline_circle|polyline_square|triangles_cube|stl_ascii_cube|stl_binary_cube>\n";

/* Unit cube [0.25, 0.75]^3: vertex b has coordinates (b & 1, b & 2, b & 4) scaled into the cube */
static const PetscInt cube_tri[36] = {0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6};

static void CubeVertex_Private(PetscInt b, PetscReal x[])
{
  x[0] = (b & 1) ? 0.75 : 0.25;
  x[1] = (b & 2) ? 0.75 : 0.25;
  x[2] = (b & 4) ? 0.75 : 0.25;
}

static PetscErrorCode CircleLS_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = PetscSqrtReal((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) - 0.3;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SquareLS_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = PetscMax(PetscAbsReal(x[0] - 0.5), PetscAbsReal(x[1] - 0.5)) - 0.25;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CubeLS_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = PetscMax(PetscMax(PetscAbsReal(x[0] - 0.5), PetscAbsReal(x[1] - 0.5)), PetscAbsReal(x[2] - 0.5)) - 0.25;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteCubeSTL_Private(const char name[], PetscBool binary)
{
  FILE         *fp;
  PetscMPIInt   rank;
  PetscInt      t, v, d;
  PetscReal     x[3];
  float         f[12];
  unsigned char header[80], count[4] = {12, 0, 0, 0}, attr[2] = {0, 0};

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  if (rank) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFOpen(PETSC_COMM_SELF, name, binary ? "wb" : "w", &fp));
  if (binary) {
    PetscCall(PetscArrayzero(header, 80));
    PetscCheck(fwrite(header, 1, 80, fp) == 80 && fwrite(count, 1, 4, fp) == 4, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Cannot write %s", name);
  } else PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "solid cube\n"));
  for (t = 0; t < 12; ++t) {
    if (binary) {
      for (d = 0; d < 3; ++d) f[d] = 0.f;
      for (v = 0; v < 3; ++v) {
        CubeVertex_Private(cube_tri[3 * t + v], x);
        for (d = 0; d < 3; ++d) f[3 + 3 * v + d] = (float)x[d];
      }
      PetscCheck(fwrite(f, sizeof(float), 12, fp) == 12 && fwrite(attr, 1, 2, fp) == 2, PETSC_COMM_SELF, PETSC_ERR_FILE_WRITE, "Cannot write %s", name);
    } else {
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "facet normal 0 0 0\nouter loop\n"));
      for (v = 0; v < 3; ++v) {
        CubeVertex_Private(cube_tri[3 * t + v], x);
        PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "vertex %g %g %g\n", (double)x[0], (double)x[1], (double)x[2]));
      }
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "endloop\nendfacet\n"));
    }
  }
  if (!binary) PetscCall(PetscFPrintf(PETSC_COMM_SELF, fp, "endsolid cube\n"));
  PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh_Private(PetscInt dim, IBBody body, Mesh *mesh)
{
  DM dm;

  PetscFunctionBeginUser;
  if (dim == 2) PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 12, 12, PETSC_DECIDE, PETSC_DECIDE, 0, 1, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  else PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, 8, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 1.));
  PetscCall(MeshCartesianCreate(dm, mesh));
  PetscCall(MeshAddBody(*mesh, body));
  PetscCall(MeshSetUp(*mesh));
  PetscCall(DMDestroy(&dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Counts per type on the surface mesh, and the number of points whose type differs from the level-set mesh */
static PetscErrorCode Compare_Private(Mesh ms, Mesh ml, DMStagStencilLocation loc, PetscBool print_map)
{
  DM              dm;
  PetscInt        dim, s[3], m[3], ne[3], e[3], i, j, k, d, cnt[4] = {0, 0, 0, 0}, sum[4];
  MeshIBPointType ts, tl;
  const char      sym[3] = {'.', 'f', '#'};
  PetscMPIInt     size;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(MeshGetDM(ms, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetCorners(dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2], &ne[0], &ne[1], &ne[2]));
  for (d = 0; d < 3; ++d) {
    if (d >= dim) {
      s[d] = 0;
      e[d] = 1;
    } else e[d] = s[d] + m[d] + (((d == 0 && loc == DMSTAG_LEFT) || (d == 1 && loc == DMSTAG_DOWN) || (d == 2 && loc == DMSTAG_BACK)) ? ne[d] : 0);
  }
  if (print_map && size == 1) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s map:\n", DMStagStencilLocations[loc]));
  for (k = s[2]; k < e[2]; ++k)
    for (j = e[1] - 1; j >= s[1]; --j) {
      if (print_map && size == 1) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  "));
      for (i = s[0]; i < e[0]; ++i) {
        PetscCall(MeshGetIBPointType(ms, loc, i, j, k, &ts));
        PetscCall(MeshGetIBPointType(ml, loc, i, j, k, &tl));
        ++cnt[ts];
        if (ts != tl) ++cnt[3];
        if (print_map && size == 1) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%c", sym[ts]));
      }
      if (print_map && size == 1) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    }
  PetscCallMPI(MPIU_Allreduce(cnt, sum, 4, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: fluid %" PetscInt_FMT ", forcing %" PetscInt_FMT ", solid %" PetscInt_FMT ", mismatches %" PetscInt_FMT "\n", DMStagStencilLocations[loc], sum[0], sum[1], sum[2], sum[3]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal Clean_Private(PetscReal v)
{
  return PetscAbsReal(v) < 1e-12 ? 0. : v;
}

/* Signed distance of the surface mesh at fixed element points: row j = 6 in 2D, the line (i, 4, 4) in 3D */
static PetscErrorCode PrintLevelSet_Private(Mesh ms, PetscInt dim)
{
  PetscMPIInt size;
  PetscInt    i, ni = dim == 2 ? 12 : 8;
  PetscReal   phi;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  if (size > 1) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "element phi at %s:", dim == 2 ? "j = 6" : "j = k = 4"));
  for (i = 0; i < ni; ++i) {
    PetscCall(MeshIBGetLevelSet_Internal(ms, DMSTAG_ELEMENT, i, dim == 2 ? 6 : 4, dim == 2 ? 0 : 4, &phi));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %.6f", (double)Clean_Private(phi)));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  IBBody          surf, ls;
  Mesh            ms, ml;
  char            cs[64] = "polyline_circle";
  PetscBool       iscircle, issquare, istri, isascii, isbinary;
  PetscInt        dim, n;
  PetscMPIInt     rank;
  PetscReal       xy[128], x[24];
  const PetscReal square[8] = {0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75};

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-case", cs, sizeof(cs), NULL));
  PetscCall(PetscStrcmp(cs, "polyline_circle", &iscircle));
  PetscCall(PetscStrcmp(cs, "polyline_square", &issquare));
  PetscCall(PetscStrcmp(cs, "triangles_cube", &istri));
  PetscCall(PetscStrcmp(cs, "stl_ascii_cube", &isascii));
  PetscCall(PetscStrcmp(cs, "stl_binary_cube", &isbinary));
  dim = (iscircle || issquare) ? 2 : 3;

  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &surf));
  PetscCall(IBBodySetType(surf, IBBODYSURFACE));
  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &ls));
  PetscCall(IBBodySetType(ls, IBBODYLEVELSET));
  if (iscircle) {
    for (n = 0; n < 64; ++n) {
      xy[2 * n]     = 0.5 + 0.3 * PetscCosReal(2. * PETSC_PI * n / 64.);
      xy[2 * n + 1] = 0.5 + 0.3 * PetscSinReal(2. * PETSC_PI * n / 64.);
    }
    PetscCall(IBBodySurfaceSetPolyline(surf, 64, xy));
    PetscCall(IBBodyLevelSetSetFunction(ls, CircleLS_Private, NULL));
  } else if (issquare) {
    PetscCall(IBBodySurfaceSetPolyline(surf, 4, square));
    PetscCall(IBBodyLevelSetSetFunction(ls, SquareLS_Private, NULL));
  } else {
    if (istri) {
      for (n = 0; n < 8; ++n) CubeVertex_Private(n, &x[3 * n]);
      PetscCall(IBBodySurfaceSetTriangles(surf, 8, x, 12, cube_tri));
    } else {
      PetscCall(WriteCubeSTL_Private(isbinary ? "mesh_ex4_cube_binary.stl" : "mesh_ex4_cube_ascii.stl", isbinary));
      PetscCall(IBBodySurfaceLoadSTL(surf, isbinary ? "mesh_ex4_cube_binary.stl" : "mesh_ex4_cube_ascii.stl"));
      PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
      if (!rank) remove(isbinary ? "mesh_ex4_cube_binary.stl" : "mesh_ex4_cube_ascii.stl");
    }
    PetscCall(IBBodyLevelSetSetFunction(ls, CubeLS_Private, NULL));
  }
  PetscCall(IBBodyView(surf, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(CreateMesh_Private(dim, surf, &ms));
  PetscCall(CreateMesh_Private(dim, ls, &ml));
  PetscCall(Compare_Private(ms, ml, DMSTAG_ELEMENT, dim == 2));
  PetscCall(Compare_Private(ms, ml, DMSTAG_LEFT, PETSC_FALSE));
  PetscCall(Compare_Private(ms, ml, DMSTAG_DOWN, PETSC_FALSE));
  if (dim == 3) PetscCall(Compare_Private(ms, ml, DMSTAG_BACK, PETSC_FALSE));
  PetscCall(PrintLevelSet_Private(ms, dim));

  PetscCall(MeshDestroy(&ml));
  PetscCall(MeshDestroy(&ms));
  PetscCall(IBBodyDestroy(&ls));
  PetscCall(IBBodyDestroy(&surf));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: polyline_circle
    nsize: 1
    args: -case polyline_circle

  test:
    suffix: polyline_square
    nsize: 1
    args: -case polyline_square

  test:
    suffix: triangles_cube
    nsize: 1
    args: -case triangles_cube

  test:
    suffix: stl_ascii_cube
    nsize: 1
    args: -case stl_ascii_cube

  test:
    suffix: stl_binary_cube
    nsize: 1
    args: -case stl_binary_cube

TEST*/
