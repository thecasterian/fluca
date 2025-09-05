#include <flucameshcart.h>
#include <flucansfsm.h>
#include <flucasys.h>
#include <petscdmstag.h>
#include <math.h>

const char *help = "Taylor-Green vortex\n";

typedef struct {
  PetscReal rho, mu;
} AppCtx;

static PetscErrorCode velocity(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar val[], void *ctx)
{
  AppCtx   *app = (AppCtx *)ctx;
  PetscReal rho = app->rho;
  PetscReal mu  = app->mu;

  val[0] = PetscSinReal(x[0]) * PetscCosReal(x[1]) * PetscExpReal(-2. * mu / rho * t);
  val[1] = -PetscCosReal(x[0]) * PetscSinReal(x[1]) * PetscExpReal(-2. * mu / rho * t);
  return PETSC_SUCCESS;
}

static PetscErrorCode GetErrorNorm(Vec actual, Vec exact, PetscReal *norm_2)
{
  Vec diff;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(exact, &diff));
  /* diff = exact - actual */
  PetscCall(VecWAXPY(diff, -1.0, actual, exact));
  PetscCall(VecNorm(diff, NORM_2, norm_2));
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mesh mesh;
  NS   ns;

  PetscReal rho         = 1.; /* Density */
  PetscReal mu          = 1.; /* Viscosity */
  PetscReal t_final     = 1.; /* Final time */
  PetscInt  nsteps      = 1;  /* Number of time steps */
  PetscBool is_periodic = PETSC_FALSE;
  PetscReal dt;

  AppCtx ctx;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-rho", &rho, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &mu, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-t_final", &t_final, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nsteps", &nsteps, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-periodic", &is_periodic, NULL));
  dt      = t_final / (PetscReal)nsteps;
  ctx.rho = rho;
  ctx.mu  = mu;

  if (is_periodic) PetscCall(MeshCartCreate2d(PETSC_COMM_WORLD, MESHCART_BOUNDARY_PERIODIC, MESHCART_BOUNDARY_PERIODIC, 8, 8, PETSC_DECIDE, PETSC_DECIDE, NULL, NULL, &mesh));
  else PetscCall(MeshCartCreate2d(PETSC_COMM_WORLD, MESHCART_BOUNDARY_NONE, MESHCART_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, NULL, NULL, &mesh));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));
  PetscCall(MeshCartSetUniformCoordinates(mesh, 0., 2. * PETSC_PI, 0., 2. * PETSC_PI, 0., 0.));

  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetType(ns, NSFSM));
  PetscCall(NSSetMesh(ns, mesh));
  PetscCall(NSSetDensity(ns, rho));
  PetscCall(NSSetViscosity(ns, mu));
  PetscCall(NSSetTimeStepSize(ns, dt));

  {
    NSBoundaryCondition bc = {
      .type         = is_periodic ? NS_BC_PERIODIC : NS_BC_VELOCITY,
      .velocity     = velocity,
      .ctx_velocity = &ctx,
    };
    PetscInt ileftb, irightb, idownb, iupb;

    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_LEFT, &ileftb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_RIGHT, &irightb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_DOWN, &idownb));
    PetscCall(MeshCartGetBoundaryIndex(mesh, MESHCART_UP, &iupb));
    PetscCall(NSSetBoundaryCondition(ns, ileftb, bc));
    PetscCall(NSSetBoundaryCondition(ns, irightb, bc));
    PetscCall(NSSetBoundaryCondition(ns, idownb, bc));
    PetscCall(NSSetBoundaryCondition(ns, iupb, bc));
  }

  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));

  {
    DM                  sdm;
    Vec                 u, v, p, Nu, Nv, Nu_prev, Nv_prev;
    DMStagStencil       row;
    PetscScalar         val;
    const PetscScalar **arrcx, **arrcy;
    PetscInt            x, y, m, n, nExtrax, nExtray;
    PetscInt            iprevc, ielemc;
    PetscInt            i, j;

    PetscCall(MeshGetScalarDM(mesh, &sdm));

    PetscCall(DMStagGetCorners(sdm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));

    PetscCall(NSFSMGetVelocity(ns, &u, &v, NULL));
    PetscCall(NSFSMGetHalfStepPressure(ns, &p));
    PetscCall(NSFSMGetConvection(ns, &Nu, &Nv, NULL));
    PetscCall(NSFSMGetPreviousConvection(ns, &Nu_prev, &Nv_prev, NULL));
    PetscCall(DMStagGetProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));

    PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_LEFT, &iprevc));
    PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_ELEMENT, &ielemc));

    /**
     * Set the exact solution of Taylor-Green vortex:
     *   u(x, y, t) = sin(x) * cos(y) * exp(-2 * nu * t)
     *   v(x, y, t) = -cos(x) * sin(y) * exp(-2 * nu * t)
     *   p(x, y, t) = (rho / 4) * [cos(2x) + cos(2y)] * exp(-4 * nu * t)
     */
    row.loc = DMSTAG_ELEMENT;
    row.c   = 0;
    for (j = y; j < y + n; ++j) {
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        /* Velocity at t = 0 */
        val = PetscSinReal(arrcx[i][ielemc]) * PetscCosReal(arrcy[j][ielemc]);
        DMStagVecSetValuesStencil(sdm, u, 1, &row, &val, INSERT_VALUES);
        val = -PetscCosReal(arrcx[i][ielemc]) * PetscSinReal(arrcy[j][ielemc]);
        DMStagVecSetValuesStencil(sdm, v, 1, &row, &val, INSERT_VALUES);
        /* Pressure at t = -dt/2 */
        val = rho / 4. * (PetscCosReal(2. * arrcx[i][ielemc]) + PetscCosReal(2. * arrcy[j][ielemc])) * PetscExpReal(2. * mu / rho * dt);
        DMStagVecSetValuesStencil(sdm, p, 1, &row, &val, INSERT_VALUES);
        /* Convection at t = 0 */
        val = 0.5 * PetscSinReal(2. * arrcx[i][ielemc]);
        DMStagVecSetValuesStencil(sdm, Nu, 1, &row, &val, INSERT_VALUES);
        val = 0.5 * PetscSinReal(2. * arrcy[j][ielemc]);
        DMStagVecSetValuesStencil(sdm, Nv, 1, &row, &val, INSERT_VALUES);
        /* Convection at t = -dt */
        val = 0.5 * PetscSinReal(2. * arrcx[i][ielemc]) * PetscExpReal(4. * mu / rho * dt);
        DMStagVecSetValuesStencil(sdm, Nu_prev, 1, &row, &val, INSERT_VALUES);
        val = 0.5 * PetscSinReal(2. * arrcy[j][ielemc]) * PetscExpReal(4. * mu / rho * dt);
        DMStagVecSetValuesStencil(sdm, Nv_prev, 1, &row, &val, INSERT_VALUES);
      }
    }

    PetscCall(DMStagRestoreProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));

    PetscCall(VecAssemblyBegin(u));
    PetscCall(VecAssemblyBegin(v));
    PetscCall(VecAssemblyBegin(p));
    PetscCall(VecAssemblyBegin(Nu));
    PetscCall(VecAssemblyBegin(Nv));
    PetscCall(VecAssemblyBegin(Nu_prev));
    PetscCall(VecAssemblyBegin(Nv_prev));
    PetscCall(VecAssemblyEnd(u));
    PetscCall(VecAssemblyEnd(v));
    PetscCall(VecAssemblyEnd(p));
    PetscCall(VecAssemblyEnd(Nu));
    PetscCall(VecAssemblyEnd(Nv));
    PetscCall(VecAssemblyEnd(Nu_prev));
    PetscCall(VecAssemblyEnd(Nv_prev));
  }

  PetscCall(NSSolve(ns, nsteps));

  {
    DM                  sdm;
    Vec                 u, v, p;
    const PetscScalar **arrcx, **arrcy;
    Vec                 u_exact, v_exact, p_exact;
    DMStagStencil       row;
    PetscScalar         val;
    PetscReal           u_norm_2, v_norm_2, p_norm_2;
    PetscInt            x, y, m, n;
    PetscInt            ielemc;
    PetscInt            i, j;

    PetscCall(MeshGetScalarDM(mesh, &sdm));
    PetscCall(DMStagGetCorners(sdm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

    PetscCall(NSFSMGetVelocity(ns, &u, &v, NULL));
    PetscCall(NSFSMGetHalfStepPressure(ns, &p));
    PetscCall(DMStagGetProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));

    PetscCall(DMCreateGlobalVector(sdm, &u_exact));
    PetscCall(DMCreateGlobalVector(sdm, &v_exact));
    PetscCall(DMCreateGlobalVector(sdm, &p_exact));

    PetscCall(DMStagGetProductCoordinateLocationSlot(sdm, DMSTAG_ELEMENT, &ielemc));

    row.loc = DMSTAG_ELEMENT;
    row.c   = 0;
    for (j = y; j < y + n; ++j) {
      for (i = x; i < x + m; ++i) {
        row.i = i;
        row.j = j;
        /* Velocity at t = t_final */
        val = PetscSinReal(arrcx[i][ielemc]) * PetscCosReal(arrcy[j][ielemc]) * PetscExpReal(-2. * mu / rho * t_final);
        DMStagVecSetValuesStencil(sdm, u_exact, 1, &row, &val, INSERT_VALUES);
        val = -PetscCosReal(arrcx[i][ielemc]) * PetscSinReal(arrcy[j][ielemc]) * PetscExpReal(-2. * mu / rho * t_final);
        DMStagVecSetValuesStencil(sdm, v_exact, 1, &row, &val, INSERT_VALUES);
        /* Pressure at t = t_final */
        val = rho / 4. * (PetscCosReal(2. * arrcx[i][ielemc]) + PetscCosReal(2. * arrcy[j][ielemc])) * PetscExpReal(-4. * mu / rho * t_final);
        DMStagVecSetValuesStencil(sdm, p_exact, 1, &row, &val, INSERT_VALUES);
      }
    }

    PetscCall(DMStagRestoreProductCoordinateArraysRead(sdm, &arrcx, &arrcy, NULL));

    PetscCall(VecAssemblyBegin(u_exact));
    PetscCall(VecAssemblyBegin(v_exact));
    PetscCall(VecAssemblyBegin(p_exact));
    PetscCall(VecAssemblyEnd(u_exact));
    PetscCall(VecAssemblyEnd(v_exact));
    PetscCall(VecAssemblyEnd(p_exact));

    PetscCall(GetErrorNorm(u, u_exact, &u_norm_2));
    PetscCall(GetErrorNorm(v, v_exact, &v_norm_2));
    PetscCall(GetErrorNorm(p, p_exact, &p_norm_2));

    PetscCall(DMRestoreGlobalVector(sdm, &u_exact));
    PetscCall(DMRestoreGlobalVector(sdm, &v_exact));
    PetscCall(DMRestoreGlobalVector(sdm, &p_exact));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 norm of u error: %g\n", u_norm_2));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 norm of v error: %g\n", v_norm_2));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L2 norm of p error: %g\n", p_norm_2));
  }

  PetscCall(MeshDestroy(&mesh));
  PetscCall(NSDestroy(&ns));

  PetscCall(FlucaFinalize());
}
