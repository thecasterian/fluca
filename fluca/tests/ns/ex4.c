#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test the immersed-boundary rows NS writes into the coupled system\n"
                           "Options:\n"
                           "  -case <jacobian_rows|function_rows|rhie_chow|nullspace> : what to print (default: jacobian_rows)\n"
                           "  -no_body : the same Phys without the disk\n"
                           "  -disk_cx <real> : x of the disk center (default: 0.625)\n";

/* Disk of radius 0.25 centered at (*cx, 0.625) */
static PetscErrorCode Disk_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  const PetscReal cx = *(const PetscReal *)ctx;

  PetscFunctionBeginUser;
  *phi = PetscSqrtReal((x[0] - cx) * (x[0] - cx) + (x[1] - 0.625) * (x[1] - 0.625)) - 0.25;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal Clean_Private(PetscReal v)
{
  return PetscAbsReal(v) < 1e-12 ? 0. : v;
}

/* Whether this rank owns the 2D stencil point (interior points only) */
static PetscErrorCode IsOwned_Private(DM dm, const DMStagStencil *st, PetscBool *owned)
{
  PetscInt xs, ys, xm, ym;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  *owned = (PetscBool)(st->i >= xs && st->i < xs + xm && st->j >= ys && st->j < ys + ym);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GlobalIndex_Private(DM dm, const DMStagStencil *st, PetscInt *g)
{
  ISLocalToGlobalMapping ltog;
  PetscInt               l;

  PetscFunctionBeginUser;
  PetscCall(DMStagStencilToIndexLocal(dm, 2, 1, st, &l));
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  PetscCall(ISLocalToGlobalMappingApply(ltog, 1, &l, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Print the nonzero entries of one row of M as (loc, i, j, c): value, scanning the star neighbourhood of width 2 */
static PetscErrorCode PrintRow_Private(DM dm, Mat M, DMStagStencil row, const char label[])
{
  const DMStagStencilLocation locs[3] = {DMSTAG_ELEMENT, DMSTAG_LEFT, DMSTAG_DOWN};
  DMStagStencil               col;
  PetscBool                   owned;
  PetscInt                    gxs, gys, gxm, gym, gr, gc, l, c, ndof, di, dj;
  PetscScalar                 v;

  PetscFunctionBeginUser;
  PetscCall(IsOwned_Private(dm, &row, &owned));
  if (owned) {
    PetscCall(DMStagGetGhostCorners(dm, &gxs, &gys, NULL, &gxm, &gym, NULL));
    PetscCall(GlobalIndex_Private(dm, &row, &gr));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%s row (%s, %" PetscInt_FMT ", %" PetscInt_FMT ", c=%" PetscInt_FMT "):\n", label, DMStagStencilLocations[row.loc], row.i, row.j, row.c));
    for (l = 0; l < 3; ++l) {
      PetscCall(DMStagGetLocationDOF(dm, locs[l], &ndof));
      for (c = 0; c < ndof; ++c)
        for (dj = -2; dj <= 2; ++dj)
          for (di = -2; di <= 2; ++di) {
            col.loc = locs[l];
            col.c   = c;
            col.i   = row.i + di;
            col.j   = row.j + dj;
            col.k   = 0;
            if (col.i < gxs || col.i >= gxs + gxm || col.j < gys || col.j >= gys + gym) continue;
            PetscCall(GlobalIndex_Private(dm, &col, &gc));
            if (gc < 0) continue;
            PetscCall(MatGetValues(M, 1, &gr, 1, &gc, &v));
            if (Clean_Private(PetscRealPart(v)) == 0.) continue;
            PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  (%s, %" PetscInt_FMT ", %" PetscInt_FMT ", c=%" PetscInt_FMT "): %.6f\n", DMStagStencilLocations[col.loc], col.i, col.j, col.c, (double)PetscRealPart(v)));
          }
    }
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrintEntry_Private(DM dm, Vec f, DMStagStencil st, const char label[])
{
  PetscBool   owned;
  PetscInt    g;
  PetscScalar v;

  PetscFunctionBeginUser;
  PetscCall(IsOwned_Private(dm, &st, &owned));
  if (owned) {
    PetscCall(GlobalIndex_Private(dm, &st, &g));
    PetscCall(VecGetValues(f, 1, &g, &v));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%s (%s, %" PetscInt_FMT ", %" PetscInt_FMT ", c=%" PetscInt_FMT "): %.6f\n", label, DMStagStencilLocations[st.loc], st.i, st.j, st.c, (double)Clean_Private(PetscRealPart(v))));
  }
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static DMStagStencil Point_Private(DMStagStencilLocation loc, PetscInt i, PetscInt j, PetscInt c)
{
  DMStagStencil st;

  st.loc = loc;
  st.i   = i;
  st.j   = j;
  st.k   = 0;
  st.c   = c;
  return st;
}

int main(int argc, char **argv)
{
  DM                 dm, sol_dm;
  Mesh               mesh;
  IBBody             body;
  Phys               phys;
  NS                 ns;
  SNES               snes;
  Mat                J, M;
  Vec                f, sub;
  IS                 is_p;
  MatNullSpace       nsp;
  PhysBC             wall      = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL};
  const PetscScalar  u_body[2] = {1., 0.};
  const PetscScalar *arr;
  char               cs[64] = "jacobian_rows";
  PetscBool          nobody = PETSC_FALSE, isjac, isfun, isrc, isnull, has_cnst;
  PetscInt           c_vel, c_U, c_p, k, n, nvec, cnt = 0;
  PetscReal          disk_cx = 0.625;
  const Vec         *vecs;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-case", cs, sizeof(cs), NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-no_body", &nobody, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-disk_cx", &disk_cx, NULL));
  PetscCall(PetscStrcmp(cs, "jacobian_rows", &isjac));
  PetscCall(PetscStrcmp(cs, "function_rows", &isfun));
  PetscCall(PetscStrcmp(cs, "rhie_chow", &isrc));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 10, 10, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1.25, 0., 1.25, 0., 0.));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  if (!nobody) {
    PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &body));
    PetscCall(IBBodySetType(body, IBBODYLEVELSET));
    PetscCall(IBBodyLevelSetSetFunction(body, Disk_Private, &disk_cx));
    PetscCall(IBBodySetVelocity(body, 2, u_body));
    PetscCall(MeshAddBody(mesh, body));
    PetscCall(IBBodyDestroy(&body));
  }
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetMesh(phys, mesh));
  PetscCall(PhysSetViscosity(phys, 0.1));
  for (k = 0; k < 4; ++k) PetscCall(PhysSetBoundaryCondition(phys, k, wall));
  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetTimeStepSize(ns, 0.05));
  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));

  /* The system before any step: zero state, so the convective part is zero */
  PetscCall(DMCreateMatrix(sol_dm, &M));
  PetscCall(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(NSFormJacobian(ns, NULL, M));
  PetscCall(DMCreateGlobalVector(sol_dm, &f));
  PetscCall(NSFormFunction(ns, NULL, f));

  if (isjac) {
    PetscCall(PrintRow_Private(sol_dm, M, Point_Private(DMSTAG_ELEMENT, 1, 1, c_vel), "Fluid momentum"));
    PetscCall(PrintRow_Private(sol_dm, M, Point_Private(DMSTAG_ELEMENT, 2, 4, c_vel), "Forcing momentum"));
    PetscCall(PrintRow_Private(sol_dm, M, Point_Private(DMSTAG_ELEMENT, 4, 4, c_vel), "Solid momentum"));
    PetscCall(PrintRow_Private(sol_dm, M, Point_Private(DMSTAG_LEFT, 3, 4, c_U), "IB face"));
    PetscCall(PrintRow_Private(sol_dm, M, Point_Private(DMSTAG_ELEMENT, 4, 4, c_p), "Solid pressure"));
    PetscCall(PrintRow_Private(sol_dm, M, Point_Private(DMSTAG_ELEMENT, 1, 1, c_p), "Fluid pressure"));
  } else if (isfun) {
    PetscCall(PrintEntry_Private(sol_dm, f, Point_Private(DMSTAG_ELEMENT, 4, 4, c_vel), "Solid momentum RHS"));
    PetscCall(PrintEntry_Private(sol_dm, f, Point_Private(DMSTAG_ELEMENT, 4, 4, c_vel + 1), "Solid momentum RHS"));
    PetscCall(PrintEntry_Private(sol_dm, f, Point_Private(DMSTAG_ELEMENT, 2, 4, c_vel), "Forcing momentum RHS"));
    PetscCall(PrintEntry_Private(sol_dm, f, Point_Private(DMSTAG_LEFT, 3, 4, c_U), "IB face RHS"));
    PetscCall(PrintEntry_Private(sol_dm, f, Point_Private(DMSTAG_ELEMENT, 4, 4, c_p), "Solid pressure RHS"));
  } else if (isrc) {
    PetscCall(PrintRow_Private(sol_dm, M, Point_Private(DMSTAG_LEFT, 2, 4, c_U), "Rhie-Chow face next to a forcing cell"));
  } else {
    PetscCall(NSGetSNES(ns, &snes));
    PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
    PetscCall(MatGetNullSpace(J, &nsp));
    PetscCall(MatNullSpaceGetVecs(nsp, &has_cnst, &nvec, &vecs));
    PetscCall(NSGetField(ns, PHYS_FIELD_PRESSURE, &is_p));
    PetscCall(VecGetSubVector(vecs[0], is_p, &sub));
    PetscCall(VecGetLocalSize(sub, &n));
    PetscCall(VecGetArrayRead(sub, &arr));
    for (k = 0; k < n; ++k)
      if (PetscRealPart(arr[k]) > 0.) ++cnt;
    PetscCall(VecRestoreArrayRead(sub, &arr));
    PetscCall(VecRestoreSubVector(vecs[0], is_p, &sub));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &cnt, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    PetscCall(MatNullSpaceTest(nsp, M, &isnull));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Null vector: nonzero pressure entries %" PetscInt_FMT ", J n = 0: %s\n", cnt, isnull ? "yes" : "no"));
  }

  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&M));
  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: no_body
    nsize: 1
    args: -case jacobian_rows -no_body

  test:
    suffix: jacobian_rows
    nsize: 1
    args: -case jacobian_rows

  test:
    suffix: function_rows
    nsize: 1
    args: -case function_rows

TEST*/
