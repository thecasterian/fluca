#include <fluca/private/flucaviewerimpl.h>
#include <petsc/private/vecimpl.h>

/*
 * Since VecLoad() does not support custom viewer type, let's define its modified version here.
 * c.f.) https://gitlab.com/petsc/petsc/-/merge_requests/8823
 */
PetscErrorCode FlucaVecLoad(Vec vec, PetscViewer viewer)
{
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(vec, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(vec, 1, viewer, 2);

  PetscCall(VecSetErrorIfLocked(vec, 1));
  if (!((PetscObject)vec)->type_name && !vec->ops->create) PetscCall(VecSetType(vec, VECSTANDARD));
  PetscCall(PetscLogEventBegin(VEC_Load, viewer, 0, 0, 0));
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_NATIVE && vec->ops->loadnative) {
    PetscUseTypeMethod(vec, loadnative, viewer);
  } else {
    PetscUseTypeMethod(vec, load, viewer);
  }
  PetscCall(PetscLogEventEnd(VEC_Load, viewer, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
