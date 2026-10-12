#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "2D flow past a cylinder in a channel (Schaefer-Turek 2D-1, Re = 20) with NS and an immersed boundary\n"
                           "Channel [0, 0.2 + L] x [0, 0.41], cylinder center (0.2, 0.2), diameter 0.1, rho = 1, nu = 1e-3.\n"
                           "Inlet and outlet: u = 4 Um y (H - y) / H^2, Um = 0.3; walls: u = 0.\n"
                           "Prints the pressure difference between the front and back points and the recirculation length.\n"
                           "Options:\n"
                           "  -outlet_length <real> : L, channel length downstream of the cylinder center (default: 2.0)\n"
                           "  -stag_grid_x <int>, -stag_grid_y <int> : grid cells (default: 440 x 82, h = 0.005 for L = 2.0)\n"
                           "  run to steady state, e.g. -ns_time_step_size 0.02 -ns_max_time 30\n";

#define H   0.41
#define UM  0.3
#define XC  0.2
#define YC  0.2
#define RAD 0.05
#define DP  0.1175
#define LA  0.0847

static PetscErrorCode Cylinder_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = PetscSqrtReal((x[0] - XC) * (x[0] - XC) + (x[1] - YC) * (x[1] - YC)) - RAD;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Poiseuille_Private(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx)
{
  PetscFunctionBeginUser;
  *val = comp == 0 ? 4. * UM * x[1] * (H - x[1]) / (H * H) : 0.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Delta p and the recirculation length along y = YC; uniform grid assumed. The two cell rows around y = YC are
   gathered on every rank and blended linearly in y. */
static PetscErrorCode Report_Private(Phys phys, Vec sol, PetscReal Lx)
{
  DM                   dm;
  Vec                  sl;
  const PetscScalar ***arr;
  PetscReal           *buf, *u, *p, hx, hy, wy, xf, xb, pf, pb, xr = -1.;
  PetscInt             xs, ys, xm, ym, Nx, Ny, i, jl, r, c_vel, c_p, su, sp, i_f, i_b;

  PetscFunctionBeginUser;
  PetscCall(PhysGetSolutionDM(phys, &dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, c_vel, &su));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, c_p, &sp));
  PetscCall(DMStagGetCorners(dm, &xs, &ys, NULL, &xm, &ym, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetGlobalSizes(dm, &Nx, &Ny, NULL));
  hx = Lx / Nx;
  hy = H / Ny;
  jl = (PetscInt)PetscFloorReal(YC / hy - 0.5);
  wy = (YC - (jl + 0.5) * hy) / hy;

  /* buf = [u row jl, u row jl + 1, p row jl, p row jl + 1], each of length Nx */
  PetscCall(PetscCalloc1(4 * Nx, &buf));
  PetscCall(DMGetLocalVector(dm, &sl));
  PetscCall(DMGlobalToLocal(dm, sol, INSERT_VALUES, sl));
  PetscCall(DMStagVecGetArrayRead(dm, sl, &arr));
  for (r = 0; r < 2; ++r) {
    if (jl + r < ys || jl + r >= ys + ym) continue;
    for (i = xs; i < xs + xm; ++i) {
      buf[r * Nx + i]       = PetscRealPart(arr[jl + r][i][su]);
      buf[(2 + r) * Nx + i] = PetscRealPart(arr[jl + r][i][sp]);
    }
  }
  PetscCall(DMStagVecRestoreArrayRead(dm, sl, &arr));
  PetscCall(DMRestoreLocalVector(dm, &sl));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, buf, 4 * Nx, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));

  /* Centerline values, overwritten into the first row of each pair */
  u = buf;
  p = buf + 2 * Nx;
  for (i = 0; i < Nx; ++i) {
    u[i] = (1. - wy) * u[i] + wy * u[Nx + i];
    p[i] = (1. - wy) * p[i] + wy * p[Nx + i];
  }

  /* Surface pressures, linearly extrapolated from the two nearest non-solid cell centers on each side */
  xf  = XC - RAD;
  xb  = XC + RAD;
  i_f = (PetscInt)PetscFloorReal(xf / hx - 0.5);
  i_b = (PetscInt)PetscCeilReal(xb / hx - 0.5);
  pf  = p[i_f] + (p[i_f] - p[i_f - 1]) * (xf - (i_f + 0.5) * hx) / hx;
  pb  = p[i_b] + (p[i_b] - p[i_b + 1]) * ((i_b + 0.5) * hx - xb) / hx;

  /* Recirculation: the first sign change of u from negative to non-negative behind the cylinder */
  for (i = i_b; i < Nx - 1; ++i) {
    if (u[i] >= 0.) break;
    if (u[i + 1] >= 0.) {
      xr = (i + 0.5) * hx + hx * (-u[i]) / (u[i + 1] - u[i]);
      break;
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Delta p              = %.4f (reference %.4f)\n", (double)(pf - pb), (double)DP));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Recirculation length = %.4f (reference %.4f)\n", (double)(xr > 0. ? xr - xb : 0.), (double)LA));
  PetscCall(PetscFree(buf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm;
  Mesh      mesh;
  IBBody    cyl;
  Phys      phys;
  NS        ns;
  Vec       sol;
  PhysBC    wall = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL}, flow = {PHYS_BC_VELOCITY, Poiseuille_Private, NULL, NULL, NULL};
  PetscReal L = 2.0;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-outlet_length", &L, NULL));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 440, 82, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., XC + L, 0., H, 0., 0.));

  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &cyl));
  PetscCall(IBBodySetType(cyl, IBBODYLEVELSET));
  PetscCall(IBBodyLevelSetSetFunction(cyl, Cylinder_Private, NULL));
  PetscCall(MeshAddBody(mesh, cyl));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));

  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetMesh(phys, mesh));
  PetscCall(PhysSetDensity(phys, 1.));
  PetscCall(PhysSetViscosity(phys, 1e-3));
  PetscCall(PhysSetBoundaryCondition(phys, 0, flow));
  PetscCall(PhysSetBoundaryCondition(phys, 1, flow));
  PetscCall(PhysSetBoundaryCondition(phys, 2, wall));
  PetscCall(PhysSetBoundaryCondition(phys, 3, wall));
  PetscCall(PhysSetFromOptions(phys));

  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSCNLINEAR));
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));

  PetscCall(NSGetSolution(ns, &sol));
  PetscCall(VecZeroEntries(sol));
  PetscCall(NSSolve(ns));
  PetscCall(Report_Private(phys, sol, XC + L));

  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(IBBodyDestroy(&cyl));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: schaefer_turek
    nsize: 1
    args: -stag_grid_x 88 -stag_grid_y 16 -ns_time_step_size 0.01 -ns_max_steps 5

TEST*/
