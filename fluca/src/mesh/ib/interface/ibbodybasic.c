#include <fluca/private/ibbodyimpl.h>

PetscClassId      IBBODY_CLASSID          = 0;
PetscFunctionList IBBodyList              = NULL;
PetscBool         IBBodyRegisterAllCalled = PETSC_FALSE;

PetscErrorCode IBBodyCreate(MPI_Comm comm, IBBody *body)
{
  IBBody   b;
  PetscInt d;

  PetscFunctionBegin;
  PetscAssertPointer(body, 2);

  PetscCall(IBBodyInitializePackage());
  PetscCall(FlucaHeaderCreate(b, IBBODY_CLASSID, "IBBody", "Immersed boundary body", "IBBody", comm, IBBodyDestroy, IBBodyView));
  for (d = 0; d < IBBODY_MAX_DIM; ++d) b->velocity[d] = 0.;
  b->vel_fn      = NULL;
  b->vel_fn_dot  = NULL;
  b->vel_ctx     = NULL;
  b->data        = NULL;
  b->setupcalled = PETSC_FALSE;
  b->sampled     = PETSC_FALSE;

  *body = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodySetType(IBBody body, IBBodyType type)
{
  IBBodyType old_type;
  PetscErrorCode (*impl_create)(IBBody);
  PetscBool match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(body, IBBODY_CLASSID, 1);
  PetscAssertPointer(type, 2);

  PetscCall(IBBodyGetType(body, &old_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)body, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(!body->sampled, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the type of a body that a Mesh has sampled");

  PetscCall(PetscFunctionListFind(IBBodyList, type, &impl_create));
  PetscCheck(impl_create, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown IBBody type: %s", type);

  if (old_type) {
    PetscTryTypeMethod(body, destroy);
    PetscCall(PetscMemzero(body->ops, sizeof(struct _IBBodyOps)));
  }

  body->setupcalled = PETSC_FALSE;
  PetscCall(PetscObjectChangeTypeName((PetscObject)body, type));
  PetscCall((*impl_create)(body));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyGetType(IBBody body, IBBodyType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(body, IBBODY_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(IBBodyRegisterAll());
  *type = ((PetscObject)body)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodySetFromOptions(IBBody body)
{
  char      type[256];
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(body, IBBODY_CLASSID, 1);
  PetscCall(IBBodyRegisterAll());

  PetscObjectOptionsBegin((PetscObject)body);
  PetscCall(PetscOptionsFList("-ibbody_type", "IBBody type", "IBBodySetType", IBBodyList, (char *)(((PetscObject)body)->type_name ? ((PetscObject)body)->type_name : IBBODYLEVELSET), type, sizeof(type), &flg));
  if (flg) PetscCall(IBBodySetType(body, type));
  else if (!((PetscObject)body)->type_name) PetscCall(IBBodySetType(body, IBBODYLEVELSET));
  PetscTryTypeMethod(body, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodySetUp(IBBody body)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(body, IBBODY_CLASSID, 1);
  if (body->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(((PetscObject)body)->type_name, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_WRONGSTATE, "IBBody type not set. Call IBBodySetType() first");
  PetscTryTypeMethod(body, setup);
  body->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyView(IBBody body, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(body, IBBODY_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)body), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(body, 1, viewer, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)body, viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    if (body->vel_fn) PetscCall(PetscViewerASCIIPrintf(viewer, "Velocity: function%s\n", body->vel_fn_dot ? " with time derivative" : ""));
    else PetscCall(PetscViewerASCIIPrintf(viewer, "Velocity: (%g, %g, %g)\n", (double)PetscRealPart(body->velocity[0]), (double)PetscRealPart(body->velocity[1]), (double)PetscRealPart(body->velocity[2])));
    PetscTryTypeMethod(body, view, viewer);
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyDestroy(IBBody *body)
{
  PetscFunctionBegin;
  if (!*body) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*body), IBBODY_CLASSID, 1);

  if (--((PetscObject)(*body))->refct > 0) {
    *body = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod((*body), destroy);
  PetscCall(PetscHeaderDestroy(body));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodySetVelocity(IBBody body, PetscInt n, const PetscScalar u[])
{
  PetscInt d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(body, IBBODY_CLASSID, 1);
  PetscCheck(n >= 0 && n <= IBBODY_MAX_DIM, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_OUTOFRANGE, "Number of velocity components %" PetscInt_FMT " out of range [0, %d]", n, IBBODY_MAX_DIM);
  if (n > 0) PetscAssertPointer(u, 3);
  for (d = 0; d < IBBODY_MAX_DIM; ++d) body->velocity[d] = d < n ? u[d] : 0.;
  body->vel_fn     = NULL;
  body->vel_fn_dot = NULL;
  body->vel_ctx    = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodySetVelocityFunction(IBBody body, IBBodyVelocityFn *fn, IBBodyVelocityFn *fn_dot, void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(body, IBBODY_CLASSID, 1);
  body->vel_fn     = fn;
  body->vel_fn_dot = fn_dot;
  body->vel_ctx    = ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sample phi on the tensor-product points (coord[0][i], coord[1][j], coord[2][k]); n[d] = 1 for d >= dim.
   phi is indexed by i + n[0] * (j + n[1] * k). */
PetscErrorCode IBBodySample_Internal(IBBody body, PetscInt dim, const PetscInt n[], const PetscReal *const coord[], PetscReal phi[])
{
  PetscFunctionBegin;
  PetscCheck(body->setupcalled, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_WRONGSTATE, "IBBody not set up. Call IBBodySetUp() first");
  PetscCheck(!body->sampled, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_WRONGSTATE, "IBBody geometry was already released by a Mesh");
  PetscUseTypeMethod(body, sample, dim, n, coord, phi);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyReleaseGeometry_Internal(IBBody body)
{
  PetscFunctionBegin;
  PetscTryTypeMethod(body, releasegeometry);
  body->sampled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyEvaluateVelocity_Internal(IBBody body, PetscInt dim, PetscReal t, const PetscReal x[], PetscBool dot, PetscScalar u[])
{
  PetscScalar up[IBBODY_MAX_DIM], um[IBBODY_MAX_DIM];
  PetscReal   h;
  PetscInt    d;

  PetscFunctionBegin;
  if (!dot) {
    if (body->vel_fn) PetscCall(body->vel_fn(dim, t, x, body->vel_ctx, u));
    else
      for (d = 0; d < dim; ++d) u[d] = body->velocity[d];
  } else if (body->vel_fn_dot) {
    PetscCall(body->vel_fn_dot(dim, t, x, body->vel_ctx, u));
  } else if (body->vel_fn) {
    /* Scale step to avoid cancellation when |t| >> 1 */
    h = IBBODY_FD_DOT_STEP * (PetscAbsReal(t) > 1. ? PetscAbsReal(t) : 1.);
    PetscCall(body->vel_fn(dim, t + h, x, body->vel_ctx, up));
    PetscCall(body->vel_fn(dim, t - h, x, body->vel_ctx, um));
    for (d = 0; d < dim; ++d) u[d] = (up[d] - um[d]) / (2. * h);
  } else {
    for (d = 0; d < dim; ++d) u[d] = 0.;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
