#include <fluca/private/nsimpl.h>
#include <petscdmstag.h>

/* ||D U|| over the pressure rows: the divergence of the face velocity */
PetscErrorCode NSComputeDivergenceNorm(NS ns, NormType type, PetscReal *nrm)
{
  DM  dm;
  IS  is_p;
  Vec div, div_p;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(ns, type, 2);
  PetscAssertPointer(nrm, 3);
  PetscCheck(ns->setupcalled, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Must call NSSetUp() before NSComputeDivergenceNorm()");
  PetscCheck(type != NORM_1_AND_2, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_OUTOFRANGE, "NORM_1_AND_2 is not supported");
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(PhysGetFieldIS(ns->phys, PHYS_FIELD_PRESSURE, &is_p));
  PetscCall(DMGetGlobalVector(dm, &div));
  PetscCall(VecZeroEntries(div));
  PetscCall(FlucaFDApply(ns->fd_D, ns->t, dm, dm, ns->sol, div));
  PetscCall(VecGetSubVector(div, is_p, &div_p));
  PetscCall(VecNorm(div_p, type, nrm));
  PetscCall(VecRestoreSubVector(div, is_p, &div_p));
  PetscCall(DMRestoreGlobalVector(dm, &div));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Local sum of 1/2 rho |u|^2 V and local maximum of dt sum_d |u_d| / h_d over the owned cells */
static PetscErrorCode ComputeCellDiagnostics_Private(NS ns, PetscReal *ke, PetscReal *cfl)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  DM                  dm;
  Vec                 sol_local;
  PetscReal           rho;
  PetscInt            dim, c_vel, xs[3], xm[3], slot_prev, i, j, k, d;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetDensity(ns->phys, &rho));
  PetscCall(DMStagGetCorners(dm, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2], NULL, NULL, NULL));
  if (dim < 3) {
    xs[2] = 0;
    xm[2] = 1;
  }
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &slot_prev));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(DMGetLocalVector(dm, &sol_local));
  PetscCall(DMGlobalToLocal(dm, ns->sol, INSERT_VALUES, sol_local));
  *ke  = 0.;
  *cfl = 0.;
  for (k = xs[2]; k < xs[2] + xm[2]; ++k) {
    for (j = xs[1]; j < xs[1] + xm[1]; ++j) {
      for (i = xs[0]; i < xs[0] + xm[0]; ++i) {
        DMStagStencil st[3];
        PetscScalar   u[3];
        PetscReal     h, vol = 1., umag2 = 0., speed = 0.;
        PetscInt      idx[3] = {i, j, k};

        for (d = 0; d < dim; ++d) {
          st[d].i   = i;
          st[d].j   = j;
          st[d].k   = k;
          st[d].loc = DMSTAG_ELEMENT;
          st[d].c   = c_vel + d;
        }
        PetscCall(DMStagVecGetValuesStencil(dm, sol_local, dim, st, u));
        for (d = 0; d < dim; ++d) {
          h = PetscRealPart(arrc[d][idx[d] + 1][slot_prev] - arrc[d][idx[d]][slot_prev]);
          vol *= h;
          umag2 += PetscRealPart(u[d] * PetscConj(u[d]));
          speed += PetscAbsScalar(u[d]) / h;
        }
        *ke += 0.5 * rho * umag2 * vol;
        *cfl = PetscMax(*cfl, ns->dt * speed);
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm, &sol_local));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* 1/2 rho sum_c |u_c|^2 V_c over the cell velocity */
PetscErrorCode NSComputeKineticEnergy(NS ns, PetscReal *ke)
{
  PetscReal ke_local, cfl_local;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(ke, 2);
  PetscCheck(ns->setupcalled, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Must call NSSetUp() before NSComputeKineticEnergy()");
  PetscCall(ComputeCellDiagnostics_Private(ns, &ke_local, &cfl_local));
  PetscCallMPI(MPIU_Allreduce(&ke_local, ke, 1, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject)ns)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* max_c dt sum_d |u_{c,d}| / h_{c,d} over the cell velocity */
PetscErrorCode NSComputeMaxCFL(NS ns, PetscReal *cfl)
{
  PetscReal ke_local, cfl_local;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscAssertPointer(cfl, 2);
  PetscCheck(ns->setupcalled, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_WRONGSTATE, "Must call NSSetUp() before NSComputeMaxCFL()");
  PetscCall(ComputeCellDiagnostics_Private(ns, &ke_local, &cfl_local));
  PetscCallMPI(MPIU_Allreduce(&cfl_local, cfl, 1, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)ns)));
  PetscFunctionReturn(PETSC_SUCCESS);
}
