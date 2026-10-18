#include <fluca/private/flucafdimpl.h>

FLUCA_EXTERN PetscErrorCode FlucaFDCreate_Derivative(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDCreate_Composition(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDCreate_Scale(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDCreate_Sum(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDCreate_SecondOrderTVD(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDCreate_IBForcing(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDCreate_IBMask(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDCreate_IBSelect(FlucaFD);

PetscErrorCode FlucaFDRegister(const char sname[], PetscErrorCode (*function)(FlucaFD))
{
  PetscFunctionBegin;
  PetscCall(FlucaFDInitializePackage());
  PetscCall(PetscFunctionListAdd(&FlucaFDList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDRegisterAll(void)
{
  PetscFunctionBegin;
  if (FlucaFDRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  FlucaFDRegisterAllCalled = PETSC_TRUE;

  PetscCall(FlucaFDRegister(FLUCAFDDERIVATIVE, FlucaFDCreate_Derivative));
  PetscCall(FlucaFDRegister(FLUCAFDCOMPOSITION, FlucaFDCreate_Composition));
  PetscCall(FlucaFDRegister(FLUCAFDSCALE, FlucaFDCreate_Scale));
  PetscCall(FlucaFDRegister(FLUCAFDSUM, FlucaFDCreate_Sum));
  PetscCall(FlucaFDRegister(FLUCAFDSECONDORDERTVD, FlucaFDCreate_SecondOrderTVD));
  PetscCall(FlucaFDRegister(FLUCAFDIBFORCING, FlucaFDCreate_IBForcing));
  PetscCall(FlucaFDRegister(FLUCAFDIBMASK, FlucaFDCreate_IBMask));
  PetscCall(FlucaFDRegister(FLUCAFDIBSELECT, FlucaFDCreate_IBSelect));
  PetscFunctionReturn(PETSC_SUCCESS);
}
