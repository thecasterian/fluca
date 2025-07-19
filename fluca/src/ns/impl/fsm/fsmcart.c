#include <fluca/private/flucaviewercgnsimpl.h>
#include <fluca/private/nsfsmimpl.h>
#include <flucameshcart.h>
#include <petscdmstag.h>

static PetscErrorCode ComputeRHSUStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSVStar2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeRHSPprime2d_Private(KSP, Vec, void *);
static PetscErrorCode ComputeOperatorsUVstar2d_Private(KSP, Mat, Mat, void *);
static PetscErrorCode ComputeOperatorPprime2d_Private(KSP, Mat, Mat, void *);

static PetscErrorCode GetBoundaryConditions2d_Private(NS ns, NSBoundaryCondition *bcleft, NSBoundaryCondition *bcright, NSBoundaryCondition *bcdown, NSBoundaryCondition *bcup)
{
  PetscInt ileftb, irightb, idownb, iupb;

  PetscFunctionBegin;
  PetscCall(MeshCartGetBoundaryIndex(ns->mesh, MESHCART_LEFT, &ileftb));
  PetscCall(MeshCartGetBoundaryIndex(ns->mesh, MESHCART_RIGHT, &irightb));
  PetscCall(MeshCartGetBoundaryIndex(ns->mesh, MESHCART_DOWN, &idownb));
  PetscCall(MeshCartGetBoundaryIndex(ns->mesh, MESHCART_UP, &iupb));
  PetscCall(NSGetBoundaryCondition(ns, ileftb, bcleft));
  PetscCall(NSGetBoundaryCondition(ns, irightb, bcright));
  PetscCall(NSGetBoundaryCondition(ns, idownb, bcdown));
  PetscCall(NSGetBoundaryCondition(ns, iupb, bcup));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateConvection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  PetscInt             M, N, x, y, m, n, nExtrax, nExtray;
  PetscScalar       ***arrNu, ***arrNv;
  const PetscScalar ***arrUV;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, iright, idown, iup, ielem, iprevc, inextc, ielemc;
  PetscScalar          hx, hy;
  PetscInt             i, j;

  Vec            u_global, v_global, u_interp_global, u_interp_local, v_interp_global, v_interp_local;
  PetscScalar ***arru_interp, ***arrv_interp;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));

  PetscCall(DMGetGlobalVector(dm, &u_global));
  PetscCall(DMGetGlobalVector(dm, &v_global));
  PetscCall(DMGetGlobalVector(fdm, &u_interp_global));
  PetscCall(DMGetLocalVector(fdm, &u_interp_local));
  PetscCall(DMGetGlobalVector(fdm, &v_interp_global));
  PetscCall(DMGetLocalVector(fdm, &v_interp_local));

  PetscCall(DMStagVecGetArray(dm, fsm->N[0], &arrNu));
  PetscCall(DMStagVecGetArray(dm, fsm->N[1], &arrNv));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_RIGHT, 0, &iright));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_UP, 0, &iup));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(DMLocalToGlobal(dm, fsm->v[0], INSERT_VALUES, u_global));
  PetscCall(DMLocalToGlobal(dm, fsm->v[1], INSERT_VALUES, v_global));

  PetscCall(MatMult(fsm->interp_v[0], u_global, u_interp_global));
  PetscCall(MatMultAdd(fsm->interp_v[1], u_global, u_interp_global, u_interp_global));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 0, ns->bcs, ns->t, ADD_VALUES, u_interp_global));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 0, ns->bcs, ns->t, ADD_VALUES, v_interp_global));
  PetscCall(MatMult(fsm->interp_v[0], v_global, v_interp_global));
  PetscCall(MatMultAdd(fsm->interp_v[1], v_global, v_interp_global, v_interp_global));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 0, 1, ns->bcs, ns->t, ADD_VALUES, v_interp_global));
  PetscCall(NSFSMComputeVelocityInterpolationOperatorBoundaryConditionVector2d_Cart_Internal(dm, fdm, 1, 1, ns->bcs, ns->t, ADD_VALUES, v_interp_global));

  PetscCall(DMGlobalToLocal(fdm, u_interp_global, INSERT_VALUES, u_interp_local));
  PetscCall(DMGlobalToLocal(fdm, v_interp_global, INSERT_VALUES, v_interp_local));

  PetscCall(DMStagVecGetArray(fdm, u_interp_local, &arru_interp));
  PetscCall(DMStagVecGetArray(fdm, v_interp_local, &arrv_interp));
  PetscCall(DMStagVecGetArrayRead(fdm, fsm->fv, &arrUV));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      hx                 = arrcx[i][inextc] - arrcx[i][iprevc];
      hy                 = arrcy[j][inextc] - arrcy[j][iprevc];
      arrNu[j][i][ielem] = (arrUV[j][i][iright] * arru_interp[j][i][iright] - arrUV[j][i][ileft] * arru_interp[j][i][ileft]) / hx //
                         + (arrUV[j][i][iup] * arru_interp[j][i][iup] - arrUV[j][i][idown] * arru_interp[j][i][idown]) / hy;
      arrNv[j][i][ielem] = (arrUV[j][i][iright] * arrv_interp[j][i][iright] - arrUV[j][i][ileft] * arrv_interp[j][i][ileft]) / hx //
                         + (arrUV[j][i][iup] * arrv_interp[j][i][iup] - arrUV[j][i][idown] * arrv_interp[j][i][idown]) / hy;
    }

  PetscCall(DMStagVecRestoreArray(dm, fsm->N[0], &arrNu));
  PetscCall(DMStagVecRestoreArray(dm, fsm->N[1], &arrNv));
  PetscCall(DMStagVecRestoreArray(fdm, u_interp_local, &arru_interp));
  PetscCall(DMStagVecRestoreArray(fdm, v_interp_local, &arrv_interp));
  PetscCall(DMStagVecRestoreArrayRead(fdm, fsm->fv, &arrUV));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMRestoreGlobalVector(dm, &u_global));
  PetscCall(DMRestoreGlobalVector(dm, &v_global));
  PetscCall(DMRestoreGlobalVector(fdm, &u_interp_global));
  PetscCall(DMRestoreLocalVector(fdm, &u_interp_local));
  PetscCall(DMRestoreGlobalVector(fdm, &v_interp_global));
  PetscCall(DMRestoreLocalVector(fdm, &v_interp_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculateIntermediateVelocity2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  PetscErrorCode (*rhs[2])(KSP, Vec, void *) = {ComputeRHSUStar2d_Private, ComputeRHSVStar2d_Private};
  PetscInt             M, N, x, y, m, n, nExtrax, nExtray;
  PetscScalar       ***arrUV_star;
  const PetscScalar ***arrp;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          U_tilde, V_tilde, dpdx, dpdy;
  PetscScalar          h1, h2;
  PetscInt             d, i, j;

  Vec            p_global, grad_p[2], u_star_global, v_star_global, u_tilde_global, u_tilde_local, v_tilde_global, v_tilde_local;
  PetscScalar ***arru_tilde, ***arrv_tilde;

  NSBoundaryCondition bcleft, bcright, bcdown, bcup;
  PetscReal           xb[2];
  PetscScalar         vb[2];

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  /* Solve for cell-centered intermediate velocity. */
  for (d = 0; d < 2; ++d) {
    Vec s;

    PetscCall(KSPSetComputeRHS(fsm->kspv[d], rhs[d], ns));
    PetscCall(KSPSetComputeOperators(fsm->kspv[d], ComputeOperatorsUVstar2d_Private, ns));
    PetscCall(KSPSolve(fsm->kspv[d], NULL, NULL));
    PetscCall(KSPGetSolution(fsm->kspv[d], &s));
    PetscCall(DMGlobalToLocal(dm, s, INSERT_VALUES, fsm->v_star[d]));
  }

  /* Calculate face-centered intermediate velocity. */
  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, &nExtrax, &nExtray, NULL));

  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &grad_p[0]));
  PetscCall(DMGetGlobalVector(dm, &grad_p[1]));
  PetscCall(DMGetGlobalVector(dm, &u_star_global));
  PetscCall(DMGetGlobalVector(dm, &v_star_global));
  PetscCall(DMGetGlobalVector(dm, &u_tilde_global));
  PetscCall(DMGetLocalVector(dm, &u_tilde_local));
  PetscCall(DMGetGlobalVector(dm, &v_tilde_global));
  PetscCall(DMGetLocalVector(dm, &v_tilde_local));

  PetscCall(DMStagVecGetArray(fdm, fsm->fv_star, &arrUV_star));
  PetscCall(DMStagVecGetArrayRead(dm, fsm->p_half, &arrp));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(GetBoundaryConditions2d_Private(ns, &bcleft, &bcright, &bcdown, &bcup));

  PetscCall(DMLocalToGlobal(dm, fsm->p_half, INSERT_VALUES, p_global));
  PetscCall(DMLocalToGlobal(dm, fsm->v_star[0], INSERT_VALUES, u_star_global));
  PetscCall(DMLocalToGlobal(dm, fsm->v_star[1], INSERT_VALUES, v_star_global));

  PetscCall(MatMult(fsm->grad_p[0], p_global, grad_p[0]));
  PetscCall(MatMult(fsm->grad_p[1], p_global, grad_p[1]));

  PetscCall(VecWAXPY(u_tilde_global, ns->dt / ns->rho, grad_p[0], u_star_global));
  PetscCall(VecWAXPY(v_tilde_global, ns->dt / ns->rho, grad_p[1], v_star_global));

  PetscCall(DMGlobalToLocal(dm, u_tilde_global, INSERT_VALUES, u_tilde_local));
  PetscCall(DMGlobalToLocal(dm, v_tilde_global, INSERT_VALUES, v_tilde_local));
  PetscCall(DMStagVecGetArray(dm, u_tilde_local, &arru_tilde));
  PetscCall(DMStagVecGetArray(dm, v_tilde_local, &arrv_tilde));

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m + nExtrax; ++i) {
      if (i == 0) {
        /* Left boundary */
        switch (bcleft.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][iprevc];
          xb[1] = arrcy[j][ielemc];
          PetscCall(bcleft.velocity(2, ns->t + ns->dt, xb, vb, bcleft.ctx_velocity));
          arrUV_star[j][i][ileft] = vb[0];
          break;
        case NS_BC_PERIODIC:
          h1                      = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
          h2                      = arrcx[i][inextc] - arrcx[i][iprevc];
          U_tilde                 = (h2 * arru_tilde[j][i - 1][ielem] + h1 * arru_tilde[j][i][ielem]) / (h1 + h2);
          dpdx                    = (arrp[j][i][ielem] - arrp[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          arrUV_star[j][i][ileft] = U_tilde - ns->dt / ns->rho * dpdx;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for left boundary: %s", NSBoundaryConditionTypes[bcleft.type]);
        }
      } else if (i == M) {
        /* Right boundary */
        switch (bcright.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][iprevc];
          xb[1] = arrcy[j][ielemc];
          PetscCall(bcright.velocity(2, ns->t + ns->dt, xb, vb, bcright.ctx_velocity));
          arrUV_star[j][i][ileft] = vb[0];
          break;
        case NS_BC_PERIODIC:
          h1                      = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
          h2                      = arrcx[i][inextc] - arrcx[i][iprevc];
          U_tilde                 = (h2 * arru_tilde[j][i - 1][ielem] + h1 * arru_tilde[j][i][ielem]) / (h1 + h2);
          dpdx                    = (arrp[j][i][ielem] - arrp[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
          arrUV_star[j][i][ileft] = U_tilde - ns->dt / ns->rho * dpdx;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for right boundary: %s", NSBoundaryConditionTypes[bcright.type]);
        }
      } else {
        h1                      = arrcx[i - 1][inextc] - arrcx[i - 1][iprevc];
        h2                      = arrcx[i][inextc] - arrcx[i][iprevc];
        U_tilde                 = (h2 * arru_tilde[j][i - 1][ielem] + h1 * arru_tilde[j][i][ielem]) / (h1 + h2);
        dpdx                    = (arrp[j][i][ielem] - arrp[j][i - 1][ielem]) / (arrcx[i][ielemc] - arrcx[i - 1][ielemc]);
        arrUV_star[j][i][ileft] = U_tilde - ns->dt / ns->rho * dpdx;
      }
    }
  for (j = y; j < y + n + nExtray; ++j)
    for (i = x; i < x + m; ++i) {
      if (j == 0) {
        /* Down boundary */
        switch (bcdown.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][iprevc];
          PetscCall(bcdown.velocity(2, ns->t + ns->dt, xb, vb, bcdown.ctx_velocity));
          arrUV_star[j][i][idown] = vb[1];
          break;
        case NS_BC_PERIODIC:
          h1                      = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
          h2                      = arrcy[j][inextc] - arrcy[j][iprevc];
          V_tilde                 = (h2 * arrv_tilde[j - 1][i][ielem] + h1 * arrv_tilde[j][i][ielem]) / (h1 + h2);
          dpdy                    = (arrp[j][i][ielem] - arrp[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          arrUV_star[j][i][idown] = V_tilde - ns->dt / ns->rho * dpdy;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for down boundary: %s", NSBoundaryConditionTypes[bcdown.type]);
        }
      } else if (j == N) {
        /* Up boundary */
        switch (bcup.type) {
        case NS_BC_VELOCITY:
          xb[0] = arrcx[i][ielemc];
          xb[1] = arrcy[j][iprevc];
          PetscCall(bcup.velocity(2, ns->t + ns->dt, xb, vb, bcup.ctx_velocity));
          arrUV_star[j][i][idown] = vb[1];
          break;
        case NS_BC_PERIODIC:
          h1                      = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
          h2                      = arrcy[j][inextc] - arrcy[j][iprevc];
          V_tilde                 = (h2 * arrv_tilde[j - 1][i][ielem] + h1 * arrv_tilde[j][i][ielem]) / (h1 + h2);
          dpdy                    = (arrp[j][i][ielem] - arrp[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
          arrUV_star[j][i][idown] = V_tilde - ns->dt / ns->rho * dpdy;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported boundary condition type for up boundary: %s", NSBoundaryConditionTypes[bcup.type]);
        }
      } else {
        h1                      = arrcy[j - 1][inextc] - arrcy[j - 1][iprevc];
        h2                      = arrcy[j][inextc] - arrcy[j][iprevc];
        V_tilde                 = (h2 * arrv_tilde[j - 1][i][ielem] + h1 * arrv_tilde[j][i][ielem]) / (h1 + h2);
        dpdy                    = (arrp[j][i][ielem] - arrp[j - 1][i][ielem]) / (arrcy[j][ielemc] - arrcy[j - 1][ielemc]);
        arrUV_star[j][i][idown] = V_tilde - ns->dt / ns->rho * dpdy;
      }
    }

  PetscCall(DMStagVecRestoreArray(dm, u_tilde_local, &arru_tilde));
  PetscCall(DMStagVecRestoreArray(dm, v_tilde_local, &arrv_tilde));
  PetscCall(DMStagVecRestoreArray(fdm, fsm->fv_star, &arrUV_star));
  PetscCall(DMStagVecRestoreArrayRead(dm, fsm->p_half, &arrp));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMLocalToLocalBegin(fdm, fsm->fv_star, INSERT_VALUES, fsm->fv_star));
  PetscCall(DMLocalToLocalEnd(fdm, fsm->fv_star, INSERT_VALUES, fsm->fv_star));

  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p[0]));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p[1]));
  PetscCall(DMRestoreGlobalVector(dm, &u_star_global));
  PetscCall(DMRestoreGlobalVector(dm, &v_star_global));
  PetscCall(DMRestoreGlobalVector(dm, &u_tilde_global));
  PetscCall(DMRestoreLocalVector(dm, &u_tilde_local));
  PetscCall(DMRestoreGlobalVector(dm, &v_tilde_global));
  PetscCall(DMRestoreLocalVector(dm, &v_tilde_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMCalculatePressureCorrection2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;
  Vec     s;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(KSPSetComputeRHS(fsm->kspp, ComputeRHSPprime2d_Private, ns));
  PetscCall(KSPSetComputeOperators(fsm->kspp, ComputeOperatorPprime2d_Private, ns));
  PetscCall(KSPSolve(fsm->kspp, NULL, NULL));
  PetscCall(KSPGetSolution(fsm->kspp, &s));
  PetscCall(DMGlobalToLocal(dm, s, INSERT_VALUES, fsm->p_prime));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSFSMUpdate2d_Cart_Internal(NS ns)
{
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  Vec u_star_global, v_star_global, fv_star_global, p_prime_global, p_global, u_global, v_global, fv_global;
  Vec grad_p_prime[2], grad_p_prime_f, lap_p_prime;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));

  PetscCall(DMGetGlobalVector(dm, &u_star_global));
  PetscCall(DMGetGlobalVector(dm, &v_star_global));
  PetscCall(DMGetGlobalVector(fdm, &fv_star_global));
  PetscCall(DMGetGlobalVector(dm, &p_prime_global));
  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &u_global));
  PetscCall(DMGetGlobalVector(dm, &v_global));
  PetscCall(DMGetGlobalVector(fdm, &fv_global));
  PetscCall(DMGetGlobalVector(dm, &grad_p_prime[0]));
  PetscCall(DMGetGlobalVector(dm, &grad_p_prime[1]));
  PetscCall(DMGetGlobalVector(fdm, &grad_p_prime_f));
  PetscCall(DMGetGlobalVector(dm, &lap_p_prime));

  PetscCall(VecCopy(fsm->p_half, fsm->p_half_prev));
  PetscCall(VecCopy(fsm->N[0], fsm->N_prev[0]));
  PetscCall(VecCopy(fsm->N[1], fsm->N_prev[1]));

  PetscCall(DMLocalToGlobal(dm, fsm->v_star[0], INSERT_VALUES, u_star_global));
  PetscCall(DMLocalToGlobal(dm, fsm->v_star[1], INSERT_VALUES, v_star_global));
  PetscCall(DMLocalToGlobal(fdm, fsm->fv_star, INSERT_VALUES, fv_star_global));
  PetscCall(DMLocalToGlobal(dm, fsm->p_prime, INSERT_VALUES, p_prime_global));
  PetscCall(DMLocalToGlobal(dm, fsm->p_half, INSERT_VALUES, p_global));

  PetscCall(MatMult(fsm->grad_p_prime[0], p_prime_global, grad_p_prime[0]));
  PetscCall(MatMult(fsm->grad_p_prime[1], p_prime_global, grad_p_prime[1]));
  PetscCall(MatMult(fsm->grad_p_prime_f, p_prime_global, grad_p_prime_f));
  PetscCall(MatMult(fsm->lap_p_prime, p_prime_global, lap_p_prime));

  PetscCall(VecWAXPY(u_global, -ns->dt / ns->rho, grad_p_prime[0], u_star_global));
  PetscCall(VecWAXPY(v_global, -ns->dt / ns->rho, grad_p_prime[1], v_star_global));
  PetscCall(VecWAXPY(fv_global, -ns->dt / ns->rho, grad_p_prime_f, fv_star_global));

  PetscCall(VecAXPY(p_global, 1., p_prime_global));
  PetscCall(VecAXPY(p_global, -0.5 * ns->mu * ns->dt / ns->rho, lap_p_prime));

  PetscCall(DMGlobalToLocal(dm, u_global, INSERT_VALUES, fsm->v[0]));
  PetscCall(DMGlobalToLocal(dm, v_global, INSERT_VALUES, fsm->v[1]));
  PetscCall(DMGlobalToLocal(fdm, fv_global, INSERT_VALUES, fsm->fv));
  PetscCall(DMGlobalToLocal(dm, p_global, INSERT_VALUES, fsm->p_half));

  PetscCall(VecAXPBYPCZ(fsm->p, 1.5, -0.5, 0, fsm->p_half, fsm->p_half_prev));

  PetscCall(DMRestoreGlobalVector(dm, &u_star_global));
  PetscCall(DMRestoreGlobalVector(dm, &v_star_global));
  PetscCall(DMRestoreGlobalVector(fdm, &fv_star_global));
  PetscCall(DMRestoreGlobalVector(dm, &p_prime_global));
  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &u_global));
  PetscCall(DMRestoreGlobalVector(dm, &v_global));
  PetscCall(DMRestoreGlobalVector(fdm, &fv_global));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p_prime[0]));
  PetscCall(DMRestoreGlobalVector(dm, &grad_p_prime[1]));
  PetscCall(DMRestoreGlobalVector(fdm, &grad_p_prime_f));
  PetscCall(DMRestoreGlobalVector(dm, &lap_p_prime));

  PetscCall(NSFSMCalculateConvection2d_Cart_Internal(ns));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSUStar2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;

  Vec u_global, p_global, Nu_global, Nu_prev_global, helm_u, dpdx, valbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(DMGetGlobalVector(dm, &u_global));
  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &Nu_global));
  PetscCall(DMGetGlobalVector(dm, &Nu_prev_global));
  PetscCall(DMGetGlobalVector(dm, &helm_u));
  PetscCall(DMGetGlobalVector(dm, &dpdx));
  PetscCall(DMGetGlobalVector(dm, &valbc));

  PetscCall(DMLocalToGlobal(dm, fsm->v[0], INSERT_VALUES, u_global));
  PetscCall(DMLocalToGlobal(dm, fsm->p_half, INSERT_VALUES, p_global));
  PetscCall(DMLocalToGlobal(dm, fsm->N[0], INSERT_VALUES, Nu_global));
  PetscCall(DMLocalToGlobal(dm, fsm->N_prev[0], INSERT_VALUES, Nu_prev_global));

  PetscCall(MatMult(fsm->helm_v, u_global, helm_u));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 0, ns->bcs, ns->t, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, helm_u));

  PetscCall(MatMult(fsm->grad_p[0], p_global, dpdx));

  PetscCall(VecSet(valbc, 0.));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 0, ns->bcs, ns->t + ns->dt, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, valbc));

  PetscCall(VecCopy(helm_u, b));
  PetscCall(VecAXPY(b, -1.5 * ns->dt, Nu_global));
  PetscCall(VecAXPY(b, 0.5 * ns->dt, Nu_prev_global));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, dpdx));
  PetscCall(VecAXPY(b, 1.0, valbc));

  PetscCall(DMRestoreGlobalVector(dm, &u_global));
  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &Nu_global));
  PetscCall(DMRestoreGlobalVector(dm, &Nu_prev_global));
  PetscCall(DMRestoreGlobalVector(dm, &helm_u));
  PetscCall(DMRestoreGlobalVector(dm, &dpdx));
  PetscCall(DMRestoreGlobalVector(dm, &valbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSVStar2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm;

  Vec v_global, p_global, Nv_global, Nv_prev_global, helm_v, dpdy, valbc;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));

  PetscCall(DMGetGlobalVector(dm, &v_global));
  PetscCall(DMGetGlobalVector(dm, &p_global));
  PetscCall(DMGetGlobalVector(dm, &Nv_global));
  PetscCall(DMGetGlobalVector(dm, &Nv_prev_global));
  PetscCall(DMGetGlobalVector(dm, &helm_v));
  PetscCall(DMGetGlobalVector(dm, &dpdy));
  PetscCall(DMGetGlobalVector(dm, &valbc));

  PetscCall(DMLocalToGlobal(dm, fsm->v[1], INSERT_VALUES, v_global));
  PetscCall(DMLocalToGlobal(dm, fsm->p_half, INSERT_VALUES, p_global));
  PetscCall(DMLocalToGlobal(dm, fsm->N[1], INSERT_VALUES, Nv_global));
  PetscCall(DMLocalToGlobal(dm, fsm->N_prev[1], INSERT_VALUES, Nv_prev_global));

  PetscCall(MatMult(fsm->helm_v, v_global, helm_v));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 1, ns->bcs, ns->t, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, helm_v));

  PetscCall(MatMult(fsm->grad_p[1], p_global, dpdy));

  PetscCall(VecSet(valbc, 0.));
  PetscCall(NSFSMComputeVelocityHelmholtzOperatorBoundaryConditionVector2d_Cart_Internal(dm, 1, ns->bcs, ns->t + ns->dt, 0.5 * ns->mu * ns->dt / ns->rho, ADD_VALUES, valbc));

  PetscCall(VecCopy(helm_v, b));
  PetscCall(VecAXPY(b, -1.5 * ns->dt, Nv_global));
  PetscCall(VecAXPY(b, 0.5 * ns->dt, Nv_prev_global));
  PetscCall(VecAXPY(b, -ns->dt / ns->rho, dpdy));
  PetscCall(VecAXPY(b, 1.0, valbc));

  PetscCall(DMRestoreGlobalVector(dm, &v_global));
  PetscCall(DMRestoreGlobalVector(dm, &p_global));
  PetscCall(DMRestoreGlobalVector(dm, &Nv_global));
  PetscCall(DMRestoreGlobalVector(dm, &Nv_prev_global));
  PetscCall(DMRestoreGlobalVector(dm, &helm_v));
  PetscCall(DMRestoreGlobalVector(dm, &dpdy));
  PetscCall(DMRestoreGlobalVector(dm, &valbc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeRHSPprime2d_Private(KSP ksp, Vec b, void *ctx)
{
  (void)ksp;

  NS      ns  = (NS)ctx;
  NS_FSM *fsm = (NS_FSM *)ns->data;
  DM      dm, fdm;

  const PetscReal rho = ns->rho;
  const PetscReal dt  = ns->dt;

  Vec fv_star = fsm->fv_star;

  MPI_Comm             comm;
  PetscInt             M, N, x, y, m, n;
  DMStagStencil        row;
  PetscScalar          valb;
  const PetscScalar ***arrUV_star;
  const PetscScalar  **arrcx, **arrcy;
  PetscInt             ileft, idown, ielem, iprevc, inextc, ielemc;
  PetscScalar          divUV_star;
  MatNullSpace         nullspace;
  PetscInt             i, j;

  NSBoundaryCondition bcleft, bcright, bcdown, bcup;

  PetscFunctionBegin;
  PetscCall(MeshGetDM(ns->mesh, &dm));
  PetscCall(MeshGetFaceDM(ns->mesh, &fdm));
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));

  PetscCall(DMStagGetGlobalSizes(dm, &M, &N, NULL));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));

  PetscCall(DMStagVecGetArrayRead(fdm, fv_star, &arrUV_star));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_LEFT, 0, &ileft));
  PetscCall(DMStagGetLocationSlot(fdm, DMSTAG_DOWN, 0, &idown));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprevc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_RIGHT, &inextc));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielemc));

  PetscCall(GetBoundaryConditions2d_Private(ns, &bcleft, &bcright, &bcdown, &bcup));

  row.loc = DMSTAG_ELEMENT;
  row.c   = 0;

  for (j = y; j < y + n; ++j)
    for (i = x; i < x + m; ++i) {
      row.i = i;
      row.j = j;

      divUV_star = (arrUV_star[j][i + 1][ileft] - arrUV_star[j][i][ileft]) / (arrcx[i][inextc] - arrcx[i][iprevc]) //
                 + (arrUV_star[j + 1][i][idown] - arrUV_star[j][i][idown]) / (arrcy[j][inextc] - arrcy[j][iprevc]);
      valb = rho / dt * divUV_star;

      PetscCall(DMStagVecSetValuesStencil(dm, b, 1, &row, &valb, INSERT_VALUES));
    }

  PetscCall(DMStagVecRestoreArrayRead(fdm, fv_star, &arrUV_star));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrcx, &arrcy, NULL));

  // TODO: below is only for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatNullSpaceRemove(nullspace, b));
  PetscCall(MatNullSpaceDestroy(&nullspace));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorsUVstar2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  (void)J;

  NS ns = (NS)ctx;
  DM dm;

  PetscFunctionBegin;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(NSFSMComputeVelocityHelmholtzOperator2d_Cart_Internal(dm, ns->bcs, 1., -0.5 * ns->mu * ns->dt / ns->rho, Jpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ComputeOperatorPprime2d_Private(KSP ksp, Mat J, Mat Jpre, void *ctx)
{
  NS           ns = (NS)ctx;
  MPI_Comm     comm;
  DM           dm;
  MatNullSpace nullspace;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)ksp, &comm));
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(NSFSMComputePressureCorrectionLaplacianOperator2d_Cart_Internal(dm, ns->bcs, Jpre));

  // TODO: below is temporary for velocity boundary conditions
  /* Remove null space. */
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullspace));
  PetscCall(MatSetNullSpace(J, nullspace));
  PetscCall(MatNullSpaceDestroy(&nullspace));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSViewSolution_FSM_Cart_Internal(NS ns, PetscViewer v)
{
  PetscBool iscgns;
  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERFLUCACGNS, &iscgns));
  if (iscgns) PetscCall(NSViewSolution_FSM_Cart_CGNS_Internal(ns, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}
