#include <fluca/private/physinsimpl.h>

static PetscErrorCode PhysRegisterFields_INS(Phys phys)
{
  PetscFunctionBegin;
  PetscCall(PhysRegisterField_Internal(phys, PHYS_FIELD_VELOCITY, PHYS_FIELD_ELEMENT, phys->dim, PHYS_EQN_MOMENTUM));
  PetscCall(PhysRegisterField_Internal(phys, PHYS_FIELD_PRESSURE, PHYS_FIELD_ELEMENT, 1, PHYS_EQN_PRESSURE));
  PetscCall(PhysRegisterField_Internal(phys, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_FACE, 1, PHYS_EQN_AUXILIARY));
  PetscCall(PhysDeclareConstantNullSpace_Internal(phys, PHYS_FIELD_PRESSURE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysSetFromOptions_INS(Phys phys, PetscOptionItems PetscOptionsObject)
{
  PetscScalar rho, mu;
  PetscReal   rho_new, mu_new;

  PetscFunctionBegin;
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_DENSITY, &rho));
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu));
  rho_new = PetscRealPart(rho);
  mu_new  = PetscRealPart(mu);
  PetscOptionsHeadBegin(PetscOptionsObject, "INS Options");
  PetscCall(PetscOptionsReal("-phys_ins_density", "Density", "PhysINSSetDensity", PetscRealPart(rho), &rho_new, NULL));
  PetscCall(PetscOptionsReal("-phys_ins_viscosity", "Dynamic viscosity", "PhysINSSetViscosity", PetscRealPart(mu), &mu_new, NULL));
  PetscOptionsHeadEnd();
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_DENSITY, rho_new));
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_VISCOSITY, mu_new));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysSetUp_INS(Phys phys)
{
  Phys_INS      *ins   = (Phys_INS *)phys->data;
  DMBoundaryType bt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PetscInt       sw, d;

  PetscFunctionBegin;
  /* The Rhie-Chow correction R = T G_c - G^st composes the four-point interpolation T with the
     three-point cell gradient. Next to a wall the interpolation is folded onto four interior cells,
     and a face row then reaches four elements away on the side the folding points into. */
  PetscCall(DMStagGetStencilWidth(phys->sol_dm, &sw));
  PetscCheck(sw >= 4, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "PhysINS requires a base DM stencil width of at least 4, got %" PetscInt_FMT, sw);
  /* Only velocity boundary conditions are supported: every non-periodic boundary needs one */
  PetscCall(DMStagGetBoundaryTypes(phys->sol_dm, &bt[0], &bt[1], &bt[2]));
  for (d = 0; d < phys->dim; ++d) {
    if (bt[d] == DM_BOUNDARY_PERIODIC) continue;
    PetscCheck(ins->bcs[2 * d].type == PHYS_INS_BC_VELOCITY && ins->bcs[2 * d + 1].type == PHYS_INS_BC_VELOCITY, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_WRONGSTATE, "PhysINS requires a velocity boundary condition on both non-periodic boundaries in direction %" PetscInt_FMT, d);
  }
  PetscCall(PhysGetField_Internal(phys, PHYS_FIELD_VELOCITY, NULL, &ins->c_vel, NULL));
  PetscCall(PhysGetField_Internal(phys, PHYS_FIELD_PRESSURE, NULL, &ins->c_p, NULL));
  PetscCall(PhysGetField_Internal(phys, PHYS_FIELD_FACE_VELOCITY, NULL, &ins->c_U, NULL));
  PetscCall(PhysINSBuildOperators_Internal(phys));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysDestroy_INS(Phys phys)
{
  PetscFunctionBegin;
  PetscCall(PhysINSDestroyOperators_Internal(phys));
  PetscCall(PetscFree(phys->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PhysView_INS(Phys phys, PetscViewer viewer)
{
  PetscBool   isascii;
  PetscScalar rho, mu;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_DENSITY, &rho));
    PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Density: %g\n", (double)PetscRealPart(rho)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Viscosity: %g\n", (double)PetscRealPart(mu)));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysCreate_INS(Phys phys)
{
  Phys_INS *ins;
  PetscInt  f;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ins));
  PetscCall(PhysRegisterProperty_Internal(phys, PHYS_PROPERTY_DENSITY, PHYS_FIELD_ELEMENT, PHYS_PROPERTY_CONSTANT));
  PetscCall(PhysRegisterProperty_Internal(phys, PHYS_PROPERTY_VISCOSITY, PHYS_FIELD_FACE, PHYS_PROPERTY_CONSTANT));
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_DENSITY, 1.));
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_VISCOSITY, 1.));

  /* Initialize BCs to NONE */
  for (f = 0; f < PHYS_INS_MAX_FACES; f++) {
    ins->bcs[f].type       = PHYS_INS_BC_NONE;
    ins->bcs[f].fn         = NULL;
    ins->bcs[f].ctx        = NULL;
    ins->bcs[f].fn_dot     = NULL;
    ins->bcs[f].fn_dot_ctx = NULL;
  }

  /* Initialize operators to NULL */
  for (f = 0; f < PHYS_INS_MAX_DIM; f++) {
    ins->fd_laplacian[f] = NULL;
    ins->fd_grad_p[f]    = NULL;
  }

  phys->data                       = ins;
  phys->ops->registerfields        = PhysRegisterFields_INS;
  phys->ops->setfromoptions        = PhysSetFromOptions_INS;
  phys->ops->setup                 = PhysSetUp_INS;
  phys->ops->destroy               = PhysDestroy_INS;
  phys->ops->view                  = PhysView_INS;
  phys->ops->computemomentumsystem = PhysComputeMomentumSystem_INS;
  phys->ops->computecouplingsystem = PhysComputeCouplingSystem_INS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Public INS-specific setters/getters ---------------------------------- */

PetscErrorCode PhysINSSetDensity(Phys phys, PetscReal rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscValidLogicalCollectiveReal(phys, rho, 2);
  PetscCheck(rho > 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Density must be positive, got %g", (double)rho);
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_DENSITY, rho));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSGetDensity(Phys phys, PetscReal *rho)
{
  PetscScalar rho_val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscAssertPointer(rho, 2);
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_DENSITY, &rho_val));
  *rho = PetscRealPart(rho_val);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSSetViscosity(Phys phys, PetscReal mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscValidLogicalCollectiveReal(phys, mu, 2);
  PetscCheck(mu >= 0, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Viscosity must be non-negative, got %g", (double)mu);
  PetscCall(PhysSetPropertyConstant_Internal(phys, PHYS_PROPERTY_VISCOSITY, mu));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSGetViscosity(Phys phys, PetscReal *mu)
{
  PetscScalar mu_val;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscAssertPointer(mu, 2);
  PetscCall(PhysGetPropertyConstant(phys, PHYS_PROPERTY_VISCOSITY, &mu_val));
  *mu = PetscRealPart(mu_val);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSSetBoundaryCondition(Phys phys, PetscInt face, PhysINSBC bc)
{
  Phys_INS *ins = (Phys_INS *)phys->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscCheck(face >= 0 && face < PHYS_INS_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, PHYS_INS_MAX_FACES);
  ins->bcs[face] = bc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PhysINSGetBoundaryCondition(Phys phys, PetscInt face, PhysINSBC *bc)
{
  Phys_INS *ins = (Phys_INS *)phys->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(phys, PHYS_CLASSID, 1);
  PetscValidHeaderSpecificType(phys, PHYS_CLASSID, 1, PHYSINS);
  PetscAssertPointer(bc, 3);
  PetscCheck(face >= 0 && face < PHYS_INS_MAX_FACES, PetscObjectComm((PetscObject)phys), PETSC_ERR_ARG_OUTOFRANGE, "Face index %" PetscInt_FMT " out of range [0, %d)", face, PHYS_INS_MAX_FACES);
  *bc = ins->bcs[face];
  PetscFunctionReturn(PETSC_SUCCESS);
}
