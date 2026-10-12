#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "2D circular Couette flow between two immersed cylinders with NS\n"
                           "Inner cylinder r = 0.3 rotates with angular velocity 1, outer cylinder r = 0.8 is fixed, domain [-1, 1]^2.\n"
                           "Exact steady solution, rho = 1:\n"
                           "  u_theta = A r + B / r, A = -R1^2 / (R2^2 - R1^2), B = R1^2 R2^2 / (R2^2 - R1^2)\n"
                           "  p = A^2 r^2 / 2 + 2 A B ln r - B^2 / (2 r^2) + C\n"
                           "Errors are reported on the band next to the bodies (FORCING cells and FLUID cells with a FORCING neighbour)\n"
                           "and on the interior (the other non-solid cells); pressure means over non-solid cells are removed.\n"
                           "Options:\n"
                           "  -stag_grid_x <int>, -stag_grid_y <int> : grid cells per direction (default: 64)\n"
                           "  -Re <real> : Reynolds number omega R1 (R2 - R1) / nu (default: 10)\n"
                           "  -flucafd_ib_interpolation_order <1|2> : IB interpolation order (default: 2)\n"
                           "  run to steady state, e.g. -ns_time_step_size 0.05 -ns_max_time 100\n";

#define R1    0.3
#define R2    0.8
#define OMEGA 1.0

static PetscErrorCode Inner_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = PetscSqrtReal(x[0] * x[0] + x[1] * x[1]) - R1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Outer_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = R2 - PetscSqrtReal(x[0] * x[0] + x[1] * x[1]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Rotation_Private(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar u[])
{
  PetscFunctionBeginUser;
  u[0] = -OMEGA * x[1];
  u[1] = OMEGA * x[0];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RotationDot_Private(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar u[])
{
  PetscFunctionBeginUser;
  u[0] = 0.;
  u[1] = 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void Exact_Private(PetscReal x, PetscReal y, PetscReal *u, PetscReal *v, PetscReal *p)
{
  const PetscReal A = -OMEGA * R1 * R1 / (R2 * R2 - R1 * R1), B = OMEGA * R1 * R1 * R2 * R2 / (R2 * R2 - R1 * R1);
  const PetscReal r = PetscSqrtReal(x * x + y * y), ut = A * r + B / r;

  *u = -ut * y / r;
  *v = ut * x / r;
  *p = A * A * r * r / 2. + 2. * A * B * PetscLogReal(r) - B * B / (2. * r * r);
}

/* -1: SOLID, 0: band (FORCING, or FLUID with a FORCING star neighbour), 1: interior */
static PetscErrorCode GetRegion_Private(Mesh mesh, PetscInt Nx, PetscInt Ny, PetscInt i, PetscInt j, PetscInt *region)
{
  const PetscInt off[4][2] = {
    {-1, 0 },
    {1,  0 },
    {0,  -1},
    {0,  1 }
  };
  MeshIBPointType type, nt;
  PetscInt        n, qi, qj;

  PetscFunctionBeginUser;
  PetscCall(MeshGetIBPointType(mesh, DMSTAG_ELEMENT, i, j, 0, &type));
  *region = type == MESH_IB_SOLID ? -1 : (type == MESH_IB_FORCING ? 0 : 1);
  if (*region != 1) PetscFunctionReturn(PETSC_SUCCESS);
  for (n = 0; n < 4; ++n) {
    qi = i + off[n][0];
    qj = j + off[n][1];
    if (qi < 0 || qi >= Nx || qj < 0 || qj >= Ny) continue;
    PetscCall(MeshGetIBPointType(mesh, DMSTAG_ELEMENT, qi, qj, 0, &nt));
    if (nt == MESH_IB_FORCING) *region = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ReportErrors_Private(Mesh mesh, Phys phys, Vec sol)
{
  const char          *names[2] = {"band", "interior"};
  DM                   dm;
  Vec                  sl;
  const PetscScalar ***arr;
  const PetscScalar  **cx, **cy;
  PetscInt             xs, ys, xm, ym, Nx, Ny, i, j, r, c_vel, c_p, su, sv, sp, sc;
  PetscReal            x, y, ue, ve, pe, eu, ep, mean[3] = {0., 0., 0.}, acc[2][5] = {{0.}}, mx[2][2] = {{0.}};

  PetscFunctionBeginUser;
  PetscCall(PhysGetSolutionDM(phys, &dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, c_vel, &su));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, c_vel + 1, &sv));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, c_p, &sp));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &sc));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &cx, &cy, NULL));
  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Ny, NULL));
  PetscCall(DMGetLocalVector(dm, &sl));
  PetscCall(DMGlobalToLocal(dm, sol, INSERT_VALUES, sl));
  PetscCall(DMStagVecGetArrayRead(dm, sl, &arr));

  /* Pressure means over the non-solid cells: computed, exact, count */
  for (j = ys; j < ys + ym; ++j)
    for (i = xs; i < xs + xm; ++i) {
      PetscCall(GetRegion_Private(mesh, Nx, Ny, i, j, &r));
      if (r < 0) continue;
      Exact_Private(PetscRealPart(cx[i][sc]), PetscRealPart(cy[j][sc]), &ue, &ve, &pe);
      mean[0] += PetscRealPart(arr[j][i][sp]);
      mean[1] += pe;
      mean[2] += 1.;
    }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, mean, 3, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));

  /* acc[r]: count, velocity L1 sum, velocity L2 sum, pressure L1 sum, pressure L2 sum; mx[r]: velocity, pressure Linf */
  for (j = ys; j < ys + ym; ++j)
    for (i = xs; i < xs + xm; ++i) {
      PetscCall(GetRegion_Private(mesh, Nx, Ny, i, j, &r));
      if (r < 0) continue;
      x = PetscRealPart(cx[i][sc]);
      y = PetscRealPart(cy[j][sc]);
      Exact_Private(x, y, &ue, &ve, &pe);
      eu = PetscSqrtReal(PetscSqr(PetscRealPart(arr[j][i][su]) - ue) + PetscSqr(PetscRealPart(arr[j][i][sv]) - ve));
      ep = PetscAbsReal((PetscRealPart(arr[j][i][sp]) - mean[0] / mean[2]) - (pe - mean[1] / mean[2]));
      acc[r][0] += 1.;
      acc[r][1] += eu;
      acc[r][2] += eu * eu;
      acc[r][3] += ep;
      acc[r][4] += ep * ep;
      mx[r][0] = PetscMax(mx[r][0], eu);
      mx[r][1] = PetscMax(mx[r][1], ep);
    }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, acc, 10, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, mx, 4, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  for (r = 0; r < 2; ++r) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Velocity error, %-8s (%5.0f cells): L1 %.4e  L2 %.4e  Linf %.4e\n", names[r], (double)acc[r][0], (double)(acc[r][1] / acc[r][0]), (double)PetscSqrtReal(acc[r][2] / acc[r][0]), (double)mx[r][0]));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Pressure error, %-8s (%5.0f cells): L1 %.4e  L2 %.4e  Linf %.4e\n", names[r], (double)acc[r][0], (double)(acc[r][3] / acc[r][0]), (double)PetscSqrtReal(acc[r][4] / acc[r][0]), (double)mx[r][1]));
  }

  PetscCall(DMStagVecRestoreArrayRead(dm, sl, &arr));
  PetscCall(DMRestoreLocalVector(dm, &sl));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &cx, &cy, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm;
  Mesh      mesh;
  IBBody    inner, outer;
  Phys      phys;
  NS        ns;
  Vec       sol;
  PhysBC    wall = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL};
  PetscReal Re   = 10.;
  PetscInt  f;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Re", &Re, NULL));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 64, 64, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, -1., 1., -1., 1., 0., 0.));

  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &inner));
  PetscCall(IBBodySetType(inner, IBBODYLEVELSET));
  PetscCall(IBBodyLevelSetSetFunction(inner, Inner_Private, NULL));
  PetscCall(IBBodySetVelocityFunction(inner, Rotation_Private, RotationDot_Private, NULL));
  PetscCall(MeshAddBody(mesh, inner));
  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &outer));
  PetscCall(IBBodySetType(outer, IBBODYLEVELSET));
  PetscCall(IBBodyLevelSetSetFunction(outer, Outer_Private, NULL));
  PetscCall(MeshAddBody(mesh, outer));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetMesh(phys, mesh));
  PetscCall(PhysSetDensity(phys, 1.));
  PetscCall(PhysSetViscosity(phys, OMEGA * R1 * (R2 - R1) / Re));
  for (f = 0; f < 4; f++) PetscCall(PhysSetBoundaryCondition(phys, f, wall));
  PetscCall(PhysSetFromOptions(phys));

  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSCNLINEAR));
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));

  PetscCall(NSGetSolution(ns, &sol));
  PetscCall(VecZeroEntries(sol));
  PetscCall(NSSolve(ns));
  PetscCall(ReportErrors_Private(mesh, phys, sol));

  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(IBBodyDestroy(&outer));
  PetscCall(IBBodyDestroy(&inner));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: couette
    nsize: 1
    args: -stag_grid_x 16 -stag_grid_y 16 -ns_time_step_size 0.05 -ns_max_steps 5

TEST*/
