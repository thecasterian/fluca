#include <fluca/private/ibbodyimpl.h>

static PetscErrorCode IBBodySample_Presampled(IBBody body, PetscInt dim, const PetscInt n[], const PetscReal *const coord[], PetscReal phi[])
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_WRONGSTATE, "A presampled body cannot be sampled; its level set lives in the Mesh it was loaded into");
}

PetscErrorCode IBBodyCreate_Presampled(IBBody body)
{
  PetscFunctionBegin;
  body->data        = NULL;
  body->ops->sample = IBBodySample_Presampled;
  PetscFunctionReturn(PETSC_SUCCESS);
}
