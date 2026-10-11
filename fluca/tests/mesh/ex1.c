#include <flucamesh.h>
#include <flucasys.h>
#include <flucaviewer.h>
#include <fluca/private/flucaviewercgnsimpl.h> /* pcgnslib and CGNSCall, to read the file back */
#include <fluca/private/meshimpl.h>            /* MeshIBGetLevelSet_Internal, MeshIBGetBodyId_Internal */
#include <petscdmstag.h>

static const char help[] = "Test Mesh: MeshCartesian on a user-provided DMStag\n"
                           "Options:\n"
                           "  -dim <int> : spatial dimension, 1, 2 or 3 (default: 2)\n"
                           "  (DMStag options such as -stag_grid_x and -stag_boundary_type_y set the grid)\n"
                           "  -roundtrip <file> : write the stretched grid to a CGNS file, reload it into a new Mesh and print it\n"
                           "  -levelset <file> : add level-set bodies, write the mesh to a CGNS file and print the LevelSet and BodyId arrays read back from CellInfo\n"
                           "  -levelset_load <file> : add level-set bodies, write the mesh to a CGNS file, load it into a new Mesh and compare the two\n"
                           "  -levelset_load_add : with -levelset_load in 2D, add a third circle to the loaded Mesh before setting it up\n";

static PetscErrorCode CreateDM_Private(PetscInt dim, DM *dm)
{
  PetscFunctionBegin;
  switch (dim) {
  case 1:
    PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 4, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, dm));
    break;
  case 2:
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, dm));
    break;
  case 3:
    PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, 4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, NULL, dm));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %" PetscInt_FMT, dim);
  }
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetUp(*dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(*dm, 0., 1., 0., 2., 0., 3.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Stretch direction 0 to x_i = (i/N)^2 on every vertex slot the rank holds, element centers at midpoints */
static PetscErrorCode StretchX_Private(DM dm)
{
  PetscScalar **arr[3] = {NULL, NULL, NULL};
  PetscInt      gx, gm, N, iprev, ielem, i;

  PetscFunctionBegin;
  PetscCall(DMStagGetGlobalSizes(dm, &N, NULL, NULL));
  PetscCall(DMStagGetGhostCorners(dm, &gx, NULL, NULL, &gm, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArrays(dm, &arr[0], &arr[1], &arr[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielem));
  for (i = gx; i < gx + gm; ++i) {
    arr[0][i][iprev] = ((PetscReal)i / N) * ((PetscReal)i / N);
    arr[0][i][ielem] = 0.5 * (((PetscReal)i / N) * ((PetscReal)i / N) + ((PetscReal)(i + 1) / N) * ((PetscReal)(i + 1) / N));
  }
  PetscCall(DMStagRestoreProductCoordinateArrays(dm, &arr[0], &arr[1], &arr[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Vertex and element-center coordinates over the ghosted range of each direction */
static PetscErrorCode PrintCoordinates_Private(Mesh mesh)
{
  DM                  dm;
  const PetscScalar **arr[3] = {NULL, NULL, NULL};
  PetscInt            dim, gx[3], gm[3], iprev, ielem, d, i;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(mesh, &dm));
  PetscCall(MeshGetDimension(mesh, &dim));
  PetscCall(DMStagGetGhostCorners(dm, &gx[0], &gx[1], &gx[2], &gm[0], &gm[1], &gm[2]));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr[0], &arr[1], &arr[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielem));
  for (d = 0; d < dim; ++d) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Direction %" PetscInt_FMT " vertices:", d));
    for (i = gx[d]; i < gx[d] + gm[d]; ++i) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %.6g", (double)PetscRealPart(arr[d][i][iprev])));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nDirection %" PetscInt_FMT " centers:", d));
    for (i = gx[d]; i < gx[d] + gm[d]; ++i) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %.6g", (double)PetscRealPart(arr[d][i][ielem])));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr[0], &arr[1], &arr[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Bodies: two circles in 2D on [0, 1] x [0, 2], one sphere in 3D on [0, 1] x [0, 2] x [0, 3] */
static const PetscReal circles[2][3] = {
  {0.3, 0.6, 0.2 },
  {0.7, 1.4, 0.25}
};
static const PetscReal sphere[4]       = {0.5, 1., 1.5, 0.6};
static const PetscReal circle_added[3] = {0.5, 1., 0.18}; /* disjoint from both circles */

static PetscErrorCode Circle_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  const PetscReal *c = (const PetscReal *)ctx;

  PetscFunctionBeginUser;
  *phi = PetscSqrtReal((x[0] - c[0]) * (x[0] - c[0]) + (x[1] - c[1]) * (x[1] - c[1])) - c[2];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Sphere_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  const PetscReal *c = (const PetscReal *)ctx;

  PetscFunctionBeginUser;
  *phi = PetscSqrtReal((x[0] - c[0]) * (x[0] - c[0]) + (x[1] - c[1]) * (x[1] - c[1]) + (x[2] - c[2]) * (x[2] - c[2])) - c[3];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AddBodies_Private(Mesh mesh, PetscInt dim)
{
  IBBody   body;
  PetscInt b, nb = dim == 2 ? 2 : 1;

  PetscFunctionBeginUser;
  for (b = 0; b < nb; ++b) {
    PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &body));
    PetscCall(IBBodySetType(body, IBBODYLEVELSET));
    if (dim == 2) PetscCall(IBBodyLevelSetSetFunction(body, Circle_Private, (void *)circles[b]));
    else PetscCall(IBBodyLevelSetSetFunction(body, Sphere_Private, (void *)sphere));
    PetscCall(MeshAddBody(mesh, body));
    PetscCall(IBBodyDestroy(&body));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Minimum phi over the bodies and its argmin (first body on ties), as Mesh computes it */
static PetscErrorCode ExactLevelSet_Private(PetscInt dim, const PetscReal x[], PetscReal *phi, PetscInt *body)
{
  PetscReal p;
  PetscInt  b, nb = dim == 2 ? 2 : 1;

  PetscFunctionBeginUser;
  for (b = 0; b < nb; ++b) {
    if (dim == 2) PetscCall(Circle_Private(dim, x, (void *)circles[b], &p));
    else PetscCall(Sphere_Private(dim, x, (void *)sphere, &p));
    if (b == 0 || p < *phi) {
      *phi  = p;
      *body = b;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal Clean_Private(PetscReal v)
{
  return PetscAbsReal(v) < 1e-12 ? 0. : v;
}

/* Point (i, j, k) of a location: face_dir < 0 for the cell centre, otherwise the face normal to face_dir */
static void PointCoordinate_Private(PetscInt dim, const PetscReal h[], const PetscInt p[], PetscInt face_dir, PetscReal x[])
{
  PetscInt d;

  for (d = 0; d < 3; ++d) x[d] = d < dim ? (p[d] + (d == face_dir ? 0. : 0.5)) * h[d] : 0.;
}

static PetscErrorCode PrintValue_Private(PetscInt dim, const PetscReal h[], const char what[], const PetscInt p[], PetscInt face_dir, PetscReal phi, PetscInt body)
{
  PetscReal x[3], ephi;
  PetscInt  ebody = -1;

  PetscFunctionBeginUser;
  PointCoordinate_Private(dim, h, p, face_dir, x);
  PetscCall(ExactLevelSet_Private(dim, x, &ephi, &ebody));
  if (dim == 2)
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s (%" PetscInt_FMT ", %" PetscInt_FMT "): LevelSet %.6f (exact %.6f), BodyId %" PetscInt_FMT " (exact %" PetscInt_FMT ")\n", what, p[0], p[1], (double)Clean_Private(phi), (double)Clean_Private(ephi), body, ebody));
  else
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  %s (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "): LevelSet %.6f (exact %.6f), BodyId %" PetscInt_FMT " (exact %" PetscInt_FMT ")\n", what, p[0], p[1], p[2], (double)Clean_Private(phi), (double)Clean_Private(ephi), body, ebody));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Read CellInfo back with serial CGNS calls (one rank) and print its fields, the face array shapes and fixed values */
static PetscErrorCode ReadLevelSet_Private(const char file[], PetscInt dim, const PetscInt M[], const PetscReal h[])
{
  int                        fn, nsols, nfields, nuser, narrays, s, f, u, a, sol = 0, rank_a, ia_phi = 0, ia_body = 0;
  char                       name[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(GridLocation_t) gloc;
  CGNS_ENUMT(DataType_t)     type;
  cgsize_t                   rmin[3] = {1, 1, 1}, rmax[3] = {1, 1, 1}, adims[3], ncell = 1, nface;
  double                    *phi;
  int                       *body;
  PetscInt                   d, p[3];
  PetscBool                  same;

  PetscFunctionBeginUser;
  CGNSCall(cg_open(file, CG_MODE_READ, &fn));
  CGNSCall(cg_nsols(fn, 1, 1, &nsols));
  for (s = 1; s <= nsols; ++s) {
    CGNSCall(cg_sol_info(fn, 1, 1, s, name, &gloc));
    PetscCall(PetscStrcmp(name, "CellInfo", &same));
    if (same) sol = s;
  }
  PetscCheck(sol, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "No CellInfo solution in %s", file);

  CGNSCall(cg_nfields(fn, 1, 1, sol, &nfields));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "CellInfo fields:"));
  for (f = 1; f <= nfields; ++f) {
    CGNSCall(cg_field_info(fn, 1, 1, sol, f, &type, name));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %s(%s)", name, cg_DataTypeName(type)));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));

  for (d = 0; d < dim; ++d) {
    rmax[d] = M[d];
    ncell *= M[d];
  }
  PetscCall(PetscMalloc1(ncell, &phi));
  PetscCall(PetscMalloc1(ncell, &body));
  CGNSCall(cg_field_read(fn, 1, 1, sol, "LevelSet", CGNS_ENUMV(RealDouble), rmin, rmax, phi));
  CGNSCall(cg_field_read(fn, 1, 1, sol, "BodyId", CGNS_ENUMV(Integer), rmin, rmax, body));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Cells:\n"));
  for (s = 0; s < 3; ++s) {
    for (d = 0; d < 3; ++d) p[d] = d < dim ? (s == 0 ? 0 : s == 1 ? M[d] / 2 : M[d] - 1) : 0;
    PetscCall(PrintValue_Private(dim, h, "cell", p, -1, phi[p[0] + M[0] * (p[1] + M[1] * p[2])], body[p[0] + M[0] * (p[1] + M[1] * p[2])]));
  }
  PetscCall(PetscFree(body));
  PetscCall(PetscFree(phi));

  CGNSCall(cg_goto(fn, 1, "Zone_t", 1, "FlowSolution_t", sol, NULL));
  CGNSCall(cg_nuser_data(&nuser));
  PetscCheck(nuser == dim, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "CellInfo has %d UserDefinedData nodes", nuser);
  for (u = 1; u <= nuser; ++u) {
    ia_phi  = 0;
    ia_body = 0;
    CGNSCall(cg_goto(fn, 1, "Zone_t", 1, "FlowSolution_t", sol, NULL));
    CGNSCall(cg_user_data_read(u, name));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s:", name));
    CGNSCall(cg_goto(fn, 1, "Zone_t", 1, "FlowSolution_t", sol, "UserDefinedData_t", u, NULL));
    CGNSCall(cg_narrays(&narrays));
    for (a = 1; a <= narrays; ++a) {
      CGNSCall(cg_array_info(a, name, &type, &rank_a, adims));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %s(%s)", name, cg_DataTypeName(type)));
      for (d = 0; d < rank_a; ++d) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s%" PetscInt_FMT, d ? " x " : " [", (PetscInt)adims[d]));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "]"));
      PetscCall(PetscStrcmp(name, "LevelSet", &same));
      if (same) ia_phi = a;
      PetscCall(PetscStrcmp(name, "BodyId", &same));
      if (same) ia_body = a;
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
    PetscCheck(ia_phi && ia_body, PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Missing LevelSet or BodyId face array");
    nface = 1;
    for (d = 0; d < dim; ++d) nface *= M[d] + (d == u - 1 ? 1 : 0);
    PetscCall(PetscMalloc1(nface, &phi));
    PetscCall(PetscMalloc1(nface, &body));
    CGNSCall(cg_array_read_as(ia_phi, CGNS_ENUMV(RealDouble), phi));
    CGNSCall(cg_array_read_as(ia_body, CGNS_ENUMV(Integer), body));
    /* The last face in the face direction (owned by the last rank), at the middle of the other directions */
    for (d = 0; d < 3; ++d) p[d] = d < dim ? (d == u - 1 ? M[d] : M[d] / 2) : 0;
    {
      PetscInt n0 = M[0] + (u == 1 ? 1 : 0), n1 = dim > 1 ? M[1] + (u == 2 ? 1 : 0) : 1, idx = p[0] + n0 * (p[1] + n1 * p[2]);

      PetscCall(PrintValue_Private(dim, h, "face", p, u - 1, phi[idx], body[idx]));
    }
    PetscCall(PetscFree(body));
    PetscCall(PetscFree(phi));
  }
  CGNSCall(cg_close(fn));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Number of bodies of a loaded mesh and their types */
static PetscErrorCode PrintBodies_Private(Mesh mesh)
{
  IBBody     body;
  IBBodyType type;
  PetscInt   nb, b;

  PetscFunctionBeginUser;
  PetscCall(MeshGetNumBodies(mesh, &nb));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Loaded mesh: %" PetscInt_FMT " bodies\n", nb));
  for (b = 0; b < nb; ++b) {
    PetscCall(MeshGetBody(mesh, b, &body));
    PetscCall(IBBodyGetType(body, &type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  body %" PetscInt_FMT ": %s\n", b, type));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Over the owned points of every location: point-type counts on both meshes, phi and body-id mismatches, and the number
   of points whose body id is 2 on the loaded mesh */
static PetscErrorCode CompareIB_Private(Mesh mesh, Mesh loaded)
{
  DM                          dm;
  PetscInt                    dim, s[3], m[3], ne[3], e[3], p[3], d, l, cnt[9], bm, bl;
  PetscReal                   pm, pl;
  MeshIBPointType             tm, tl;
  const DMStagStencilLocation locs[4] = {DMSTAG_ELEMENT, DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

  PetscFunctionBeginUser;
  PetscCall(MeshGetDM(mesh, &dm));
  PetscCall(MeshGetDimension(mesh, &dim));
  PetscCall(DMStagGetCorners(dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2], &ne[0], &ne[1], &ne[2]));
  for (d = dim; d < 3; ++d) {
    s[d] = 0;
    m[d] = 1;
  }
  for (l = 0; l <= dim; ++l) {
    for (d = 0; d < 3; ++d) e[d] = s[d] + m[d] + (d < dim && l == d + 1 ? ne[d] : 0);
    /* cnt: original FLUID, FORCING, SOLID, loaded FLUID, FORCING, SOLID, phi mismatches, body-id mismatches, body-2 points */
    for (d = 0; d < 9; ++d) cnt[d] = 0;
    for (p[2] = s[2]; p[2] < e[2]; ++p[2])
      for (p[1] = s[1]; p[1] < e[1]; ++p[1])
        for (p[0] = s[0]; p[0] < e[0]; ++p[0]) {
          PetscCall(MeshGetIBPointType(mesh, locs[l], p[0], p[1], p[2], &tm));
          PetscCall(MeshGetIBPointType(loaded, locs[l], p[0], p[1], p[2], &tl));
          PetscCall(MeshIBGetLevelSet_Internal(mesh, locs[l], p[0], p[1], p[2], &pm));
          PetscCall(MeshIBGetLevelSet_Internal(loaded, locs[l], p[0], p[1], p[2], &pl));
          PetscCall(MeshIBGetBodyId_Internal(mesh, locs[l], p[0], p[1], p[2], &bm));
          PetscCall(MeshIBGetBodyId_Internal(loaded, locs[l], p[0], p[1], p[2], &bl));
          ++cnt[tm];
          ++cnt[3 + tl];
          if (PetscAbsReal(pm - pl) > 1e-14) ++cnt[6];
          if (bm != bl) ++cnt[7];
          if (bl == 2) ++cnt[8];
        }
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, cnt, 9, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: original FLUID %" PetscInt_FMT " FORCING %" PetscInt_FMT " SOLID %" PetscInt_FMT ", loaded FLUID %" PetscInt_FMT " FORCING %" PetscInt_FMT " SOLID %" PetscInt_FMT "\n", DMStagStencilLocations[locs[l]], cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "  LevelSet mismatches %" PetscInt_FMT ", BodyId mismatches %" PetscInt_FMT ", body 2 points %" PetscInt_FMT "\n", cnt[6], cnt[7], cnt[8]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  Mesh        mesh, loaded;
  PetscInt    dim = 2;
  char        file[PETSC_MAX_PATH_LEN];
  PetscBool   roundtrip;
  PetscViewer viewer;
  char        lsfile[PETSC_MAX_PATH_LEN];
  PetscBool   levelset;
  char        ldfile[PETSC_MAX_PATH_LEN];
  PetscBool   levelset_load, levelset_load_add = PETSC_FALSE;
  PetscInt    M[3] = {1, 1, 1};
  PetscReal   h[3];

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-roundtrip", file, sizeof(file), &roundtrip));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-levelset", lsfile, sizeof(lsfile), &levelset));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-levelset_load", ldfile, sizeof(ldfile), &levelset_load));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-levelset_load_add", &levelset_load_add, NULL));
  PetscCheck(!levelset_load_add || (levelset_load && dim == 2), PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "-levelset_load_add needs -levelset_load in 2D");

  PetscCall(CreateDM_Private(dim, &dm));
  if (roundtrip) PetscCall(StretchX_Private(dm));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  if (levelset || levelset_load) PetscCall(AddBodies_Private(mesh, dim));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));
  if (levelset) {
    PetscCall(DMStagGetGlobalSizes(dm, &M[0], &M[1], &M[2]));
    h[0] = 1. / M[0];
    h[1] = 2. / M[1];
    h[2] = 3. / M[2];
    PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, lsfile, FILE_MODE_WRITE, &viewer));
    PetscCall(MeshView(mesh, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(ReadLevelSet_Private(lsfile, dim, M, h));
  } else if (levelset_load) {
    PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, ldfile, FILE_MODE_WRITE, &viewer));
    PetscCall(MeshView(mesh, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(MeshCreate(PETSC_COMM_WORLD, &loaded));
    PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, ldfile, FILE_MODE_READ, &viewer));
    PetscCall(MeshLoad(loaded, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    if (levelset_load_add) {
      IBBody body;

      PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &body));
      PetscCall(IBBodySetType(body, IBBODYLEVELSET));
      PetscCall(IBBodyLevelSetSetFunction(body, Circle_Private, (void *)circle_added));
      PetscCall(MeshAddBody(loaded, body));
      PetscCall(IBBodyDestroy(&body));
    }
    PetscCall(MeshSetUp(loaded));
    PetscCall(PrintBodies_Private(loaded));
    PetscCall(CompareIB_Private(mesh, loaded));
    PetscCall(MeshDestroy(&loaded));
  } else if (!roundtrip) PetscCall(MeshView(mesh, PETSC_VIEWER_STDOUT_WORLD));
  else {
    PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, file, FILE_MODE_WRITE, &viewer));
    PetscCall(MeshView(mesh, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(MeshCreate(PETSC_COMM_WORLD, &loaded));
    PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
    PetscCall(MeshLoad(loaded, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(MeshSetUp(loaded));
    PetscCall(MeshView(loaded, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PrintCoordinates_Private(loaded));
    PetscCall(MeshDestroy(&loaded));
  }

  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: view_ascii_1d
    nsize: 1
    args: -dim 1 -stag_grid_x 8

  test:
    suffix: view_ascii_2d
    nsize: 1
    args: -dim 2 -stag_grid_x 8 -stag_grid_y 4 -stag_boundary_type_y periodic

  test:
    suffix: view_ascii_3d
    nsize: 1
    args: -dim 3 -stag_grid_x 4 -stag_grid_y 3 -stag_grid_z 2 -stag_boundary_type_z periodic

  test:
    suffix: cgns_roundtrip_2d
    nsize: 1
    args: -dim 2 -stag_grid_x 4 -stag_grid_y 3 -stag_boundary_type_y periodic -roundtrip mesh_ex1_2d.cgns

  test:
    suffix: cgns_roundtrip_3d
    nsize: 1
    args: -dim 3 -stag_grid_x 3 -stag_grid_y 2 -stag_grid_z 2 -stag_boundary_type_x periodic -roundtrip mesh_ex1_3d.cgns

  test:
    suffix: cgns_levelset_2d
    nsize: 1
    args: -dim 2 -stag_grid_x 8 -stag_grid_y 6 -levelset mesh_ex1_ls_2d.cgns

  test:
    suffix: cgns_levelset_3d
    nsize: 1
    args: -dim 3 -stag_grid_x 4 -stag_grid_y 4 -stag_grid_z 4 -levelset mesh_ex1_ls_3d.cgns

  test:
    suffix: cgns_levelset_load_2d
    nsize: 1
    args: -dim 2 -stag_grid_x 8 -stag_grid_y 6 -levelset_load mesh_ex1_ld_2d.cgns

  test:
    suffix: cgns_levelset_load_3d
    nsize: 1
    args: -dim 3 -stag_grid_x 4 -stag_grid_y 4 -stag_grid_z 4 -levelset_load mesh_ex1_ld_3d.cgns

  test:
    suffix: cgns_levelset_load_add_2d
    nsize: 1
    args: -dim 2 -stag_grid_x 8 -stag_grid_y 6 -levelset_load mesh_ex1_ld_add_2d.cgns -levelset_load_add

  test:
    suffix: cgns_levelset_load_periodic_2d
    nsize: 1
    args: -dim 2 -stag_grid_x 4 -stag_grid_y 6 -stag_boundary_type_x periodic -levelset_load mesh_ex1_ld_per_2d.cgns

TEST*/
