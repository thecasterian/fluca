#pragma once

#include <flucaphys.h>
#include <flucaseg.h>
#include <flucasys.h>
#include <petscdmstag.h>

/* Two-phase setup of a PhysLaminar on dm: the Phys freezes its declarations, then a SEGCNLINEAR
   adds its auxiliary fields and triggers creation of the solution DM. Every non-periodic boundary
   gets a velocity BC from bcfn (NULL means zero velocity), set before PhysSetUp() as required.
   Both objects are returned; destroy seg before phys. */
static PetscErrorCode PhysTestSetUp(DM dm, PetscReal rho, PetscReal mu, PhysLaminarBCFn *bcfn, Phys *phys, Seg *seg)
{
  DMBoundaryType bt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PhysLaminarBC  bc;
  PetscInt       dim, d;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetBoundaryTypes(dm, &bt[0], &bt[1], &bt[2]));
  PetscCall(PhysCreate(PetscObjectComm((PetscObject)dm), phys));
  PetscCall(PhysSetType(*phys, PHYSLAMINAR));
  PetscCall(PhysSetBaseDM(*phys, dm));
  PetscCall(PhysLaminarSetDensity(*phys, rho));
  PetscCall(PhysLaminarSetViscosity(*phys, mu));
  bc.type       = PHYS_LAMINAR_BC_VELOCITY;
  bc.fn         = bcfn;
  bc.ctx        = NULL;
  bc.fn_dot     = NULL;
  bc.fn_dot_ctx = NULL;
  for (d = 0; d < dim; ++d) {
    if (bt[d] == DM_BOUNDARY_PERIODIC) continue;
    PetscCall(PhysLaminarSetBoundaryCondition(*phys, 2 * d, bc));
    PetscCall(PhysLaminarSetBoundaryCondition(*phys, 2 * d + 1, bc));
  }
  PetscCall(PhysSetFromOptions(*phys));
  PetscCall(PhysSetUp(*phys));
  PetscCall(SegCreate(PetscObjectComm((PetscObject)dm), seg));
  PetscCall(SegSetType(*seg, SEGCNLINEAR));
  PetscCall(SegSetPhys(*seg, *phys));
  PetscCall(SegSetUp(*seg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Matrix and right-hand side of the coupled system (13) on the solution DM */
static PetscErrorCode PhysTestCreateSystem(Phys phys, Mat *M, Vec *f)
{
  DM sol_dm;

  PetscFunctionBeginUser;
  PetscCall(PhysGetSolutionDM(phys, &sol_dm));
  PetscCall(DMCreateMatrix(sol_dm, M));
  PetscCall(MatSetOption(*M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(*M, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(DMCreateGlobalVector(sol_dm, f));
  PetscCall(VecZeroEntries(*f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check max |a - b| over the entries of one field */
static PetscErrorCode PhysTestCheckField(Phys phys, const char name[], Vec a, Vec b, PetscReal tol)
{
  IS        is;
  Vec       sa, sb, diff;
  PetscReal err;

  PetscFunctionBeginUser;
  PetscCall(PhysGetFieldIS(phys, name, &is));
  PetscCall(VecGetSubVector(a, is, &sa));
  PetscCall(VecGetSubVector(b, is, &sb));
  PetscCall(VecDuplicate(sa, &diff));
  PetscCall(VecWAXPY(diff, -1., sb, sa));
  PetscCall(VecNorm(diff, NORM_INFINITY, &err));
  PetscCall(VecDestroy(&diff));
  PetscCall(VecRestoreSubVector(b, is, &sb));
  PetscCall(VecRestoreSubVector(a, is, &sa));
  PetscCall(ISDestroy(&is));
  PetscCheck(err <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Field %s: max error %g exceeds tolerance %g", name, (double)err, (double)tol);
  PetscFunctionReturn(PETSC_SUCCESS);
}
