#include <fluca/private/ibbodyimpl.h>

static PetscErrorCode IBBodySetUp_LevelSet(IBBody body)
{
  IBBody_LevelSet *ls = (IBBody_LevelSet *)body->data;

  PetscFunctionBegin;
  PetscCheck(ls->fn, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_WRONGSTATE, "Level-set function not set. Call IBBodyLevelSetSetFunction() first");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IBBodySample_LevelSet(IBBody body, PetscInt dim, const PetscInt n[], const PetscReal *const coord[], PetscReal phi[])
{
  IBBody_LevelSet *ls                = (IBBody_LevelSet *)body->data;
  PetscReal        x[IBBODY_MAX_DIM] = {0., 0., 0.};
  PetscInt         i, j, k;

  PetscFunctionBegin;
  for (k = 0; k < n[2]; ++k) {
    if (dim > 2) x[2] = coord[2][k];
    for (j = 0; j < n[1]; ++j) {
      if (dim > 1) x[1] = coord[1][j];
      for (i = 0; i < n[0]; ++i) {
        x[0] = coord[0][i];
        PetscCall(ls->fn(dim, x, ls->ctx, &phi[(k * n[1] + j) * n[0] + i]));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IBBodyDestroy_LevelSet(IBBody body)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(body->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyCreate_LevelSet(IBBody body)
{
  IBBody_LevelSet *ls;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ls));
  ls->fn  = NULL;
  ls->ctx = NULL;

  body->data         = (void *)ls;
  body->ops->setup   = IBBodySetUp_LevelSet;
  body->ops->sample  = IBBodySample_LevelSet;
  body->ops->destroy = IBBodyDestroy_LevelSet;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyLevelSetSetFunction(IBBody body, IBBodyLevelSetFn *fn, void *ctx)
{
  IBBody_LevelSet *ls;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(body, IBBODY_CLASSID, 1, IBBODYLEVELSET);
  PetscCheck(!body->sampled, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the geometry of a body that a Mesh has sampled");
  ls                = (IBBody_LevelSet *)body->data;
  ls->fn            = fn;
  ls->ctx           = ctx;
  body->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
