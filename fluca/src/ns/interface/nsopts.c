#include <fluca/private/nsimpl.h>

PetscErrorCode NSSetMesh(NS ns, Mesh mesh)
{
  PetscInt nb;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 2);
  PetscCheckSameComm(ns, 1, mesh, 2);
  if (ns->mesh == mesh) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(MeshDestroy(&ns->mesh));
  PetscCall(PetscFree(ns->bcs));

  ns->mesh = mesh;
  PetscCall(PetscObjectReference((PetscObject)mesh));
  PetscCall(MeshGetNumberBoundaries(mesh, &nb));
  if (nb > 0) PetscCall(PetscCalloc1(nb, &ns->bcs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetMesh(NS ns, Mesh *mesh)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (mesh) *mesh = ns->mesh;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetDensity(NS ns, PetscReal rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->rho = rho;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetDensity(NS ns, PetscReal *rho)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (rho) *rho = ns->rho;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetViscosity(NS ns, PetscReal mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->mu = mu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetViscosity(NS ns, PetscReal *mu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (mu) *mu = ns->mu;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetTimeStepSize(NS ns, PetscReal dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->dt = dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetTimeStepSize(NS ns, PetscReal *dt)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (dt) *dt = ns->dt;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetTimeStep(NS ns, PetscInt step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->step = step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetTimeStep(NS ns, PetscInt *step)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (step) *step = ns->step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetTime(NS ns, PetscReal t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  ns->t = t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetTime(NS ns, PetscReal *t)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  if (t) *t = ns->t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetBoundaryCondition(NS ns, PetscInt index, NSBoundaryCondition bc)
{
  PetscInt nb;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(MeshGetNumberBoundaries(ns->mesh, &nb));
  PetscCheck(0 <= index && index < nb, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary index %" PetscInt_FMT ", must be in [0, %" PetscInt_FMT ")", index, nb);
  ns->bcs[index] = bc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSGetBoundaryCondition(NS ns, PetscInt index, NSBoundaryCondition *bc)
{
  PetscInt nb;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(MeshGetNumberBoundaries(ns->mesh, &nb));
  PetscCheck(0 <= index && index < nb, PetscObjectComm((PetscObject)ns), PETSC_ERR_ARG_OUTOFRANGE, "Invalid boundary index %" PetscInt_FMT ", must be in [0, %" PetscInt_FMT ")", index, nb);
  if (bc) *bc = ns->bcs[index];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSetFromOptions(NS ns)
{
  char      type[256];
  PetscBool flg, opt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ns, NS_CLASSID, 1);
  PetscCall(NSRegisterAll());

  PetscObjectOptionsBegin((PetscObject)ns);

  PetscCall(PetscOptionsFList("-ns_type", "NS type", "NSSetType", NSList, (char *)(((PetscObject)ns)->type_name ? ((PetscObject)ns)->type_name : NSFSM), type, sizeof(type), &flg));
  if (flg) PetscCall(NSSetType(ns, type));
  else if (!((PetscObject)ns)->type_name) PetscCall(NSSetType(ns, NSFSM));

  PetscCall(PetscOptionsReal("-ns_density", "Fluid density", "NSSetDensity", ns->rho, &ns->rho, NULL));
  PetscCall(PetscOptionsReal("-ns_viscosity", "Fluid viscosity", "NSSetViscosity", ns->mu, &ns->mu, NULL));
  PetscCall(PetscOptionsReal("-ns_time_step_size", "Time step size", "NSSetTimeStepSize", ns->dt, &ns->dt, NULL));

  PetscCall(PetscOptionsInt("-ns_monitor_frequency", "Monitor frequency", "NSMonitorSetFrequency", ns->mon_freq, &ns->mon_freq, NULL));
  PetscCall(NSMonitorSetFromOptions(ns, "-ns_monitor", "Monitor current step and time", "NSMonitorDefault", NSMonitorDefault, NULL));
  PetscCall(NSMonitorSetFromOptions(ns, "-ns_monitor_solution", "Monitor solution", "NSMonitorSolution", NSMonitorSolution, NULL));
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-ns_monitor_cancel", "Remove all monitors", "NSMonitorCancel", flg, &flg, &opt));
  if (opt && flg) PetscCall(NSMonitorCancel(ns));

  PetscTryTypeMethod(ns, setfromoptions, PetscOptionsObject);

  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}
