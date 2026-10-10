#include <fluca/private/meshimpl.h>

const char *MeshIBPointTypes[] = {"FLUID", "FORCING", "SOLID", "MeshIBPointType", "MESH_IB_", NULL};

static PetscErrorCode MeshIBFindLocation_Private(Mesh mesh, DMStagStencilLocation loc, const MeshIBLocation **l)
{
  PetscInt n;

  PetscFunctionBegin;
  *l = NULL;
  for (n = 0; n < mesh->ib.nloc; ++n)
    if (mesh->ib.locs[n].loc == loc) *l = &mesh->ib.locs[n];
  PetscCheck(*l, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_OUTOFRANGE, "Location %s is not classified by the immersed boundary", DMStagStencilLocations[loc]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshIBGetLocalIndex_Internal(Mesh mesh, const MeshIBLocation *l, PetscInt i, PetscInt j, PetscInt k, PetscInt *idx)
{
  const MeshIB *ib = &mesh->ib;
  PetscInt      p[3], d;

  PetscFunctionBegin;
  p[0] = i;
  p[1] = j;
  p[2] = k;
  for (d = 0; d < 3; ++d) PetscCheck(ib->gx[d] <= p[d] && p[d] < ib->gx[d] + ib->gn[d], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") is outside the local ghosted range", i, j, k);
  *idx = (((p[2] - ib->gx[2]) * ib->gn[1] + (p[1] - ib->gx[1])) * ib->gn[0] + (p[0] - ib->gx[0])) * ib->epe + l->slot;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshIBIsOnGrid_Internal(Mesh mesh, DMStagStencilLocation loc, PetscInt dir, PetscInt idx, PetscBool *on)
{
  PetscFunctionBegin;
  if (dir >= mesh->dim) *on = (PetscBool)(idx == 0);
  else if (mesh->ib.periodic[dir]) *on = PETSC_TRUE;
  else *on = (PetscBool)(0 <= idx && idx < mesh->ib.N[dir] + (MeshIBIsFaceInDir_Private(loc, dir) ? 1 : 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* SOLID if phi < 0; FORCING if not solid and an on-grid star neighbour inside the ghosted range is solid */
PetscErrorCode MeshIBClassify_Internal(Mesh mesh, const MeshIBLocation *l, PetscInt i, PetscInt j, PetscInt k, MeshIBPointType *type)
{
  const MeshIB *ib = &mesh->ib;
  PetscInt      p[3], q[3], d, s, idx;
  PetscBool     on;

  PetscFunctionBegin;
  PetscCall(MeshIBGetLocalIndex_Internal(mesh, l, i, j, k, &idx));
  if (PetscRealPart(ib->arr_phi[idx]) < 0.) {
    *type = MESH_IB_SOLID;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  *type = MESH_IB_FLUID;
  p[0]  = i;
  p[1]  = j;
  p[2]  = k;
  for (d = 0; d < mesh->dim; ++d) {
    for (s = -1; s <= 1; s += 2) {
      q[0] = p[0];
      q[1] = p[1];
      q[2] = p[2];
      q[d] += s;
      PetscCall(MeshIBIsOnGrid_Internal(mesh, l->loc, d, q[d], &on));
      /* Neighbours outside the ghosted range are not examined (see MeshGetIBPointType) */
      if (!on || q[d] < ib->gx[d] || q[d] >= ib->gx[d] + ib->gn[d]) continue;
      PetscCall(MeshIBGetLocalIndex_Internal(mesh, l, q[0], q[1], q[2], &idx));
      if (PetscRealPart(ib->arr_phi[idx]) < 0.) {
        *type = MESH_IB_FORCING;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshAddBody(Mesh mesh, IBBody body)
{
  IBBody  *bodies;
  PetscInt n;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscValidHeaderSpecific(body, IBBODY_CLASSID, 2);
  PetscCheckSameComm(mesh, 1, body, 2);
  PetscCheck(!mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Cannot add a body after MeshSetUp()");
  PetscCheck(!body->sampled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "The body was already sampled by another Mesh");
  PetscCheck(mesh->ib.nbodies < MESH_IB_MAX_BODIES, PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "At most %d bodies are supported", MESH_IB_MAX_BODIES);
  PetscCall(PetscMalloc1(mesh->ib.nbodies + 1, &bodies));
  for (n = 0; n < mesh->ib.nbodies; ++n) bodies[n] = mesh->ib.bodies[n];
  bodies[mesh->ib.nbodies] = body;
  PetscCall(PetscObjectReference((PetscObject)body));
  PetscCall(PetscFree(mesh->ib.bodies));
  mesh->ib.bodies = bodies;
  ++mesh->ib.nbodies;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetNumBodies(Mesh mesh, PetscInt *n)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(n, 2);
  *n = mesh->ib.nbodies;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetBody(Mesh mesh, PetscInt n, IBBody *body)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(body, 3);
  PetscCheck(n >= 0 && n < mesh->ib.nbodies, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_OUTOFRANGE, "Body %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", n, mesh->ib.nbodies);
  *body = mesh->ib.bodies[n];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetIBLevelSet(Mesh mesh, Vec *phi)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(phi, 2);
  PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Mesh not set up. Call MeshSetUp() first");
  *phi = mesh->ib.phi_global;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshGetIBPointType(Mesh mesh, DMStagStencilLocation loc, PetscInt i, PetscInt j, PetscInt k, MeshIBPointType *type)
{
  const MeshIBLocation *l;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(type, 6);
  PetscCheck(mesh->setupcalled, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONGSTATE, "Mesh not set up. Call MeshSetUp() first");
  *type = MESH_IB_FLUID;
  if (!mesh->ib.nbodies) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MeshIBFindLocation_Private(mesh, loc, &l));
  PetscCall(MeshIBClassify_Internal(mesh, l, i, j, k, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshIBGetLevelSet_Internal(Mesh mesh, DMStagStencilLocation loc, PetscInt i, PetscInt j, PetscInt k, PetscReal *phi)
{
  const MeshIBLocation *l;
  PetscInt              idx;

  PetscFunctionBegin;
  PetscCall(MeshIBFindLocation_Private(mesh, loc, &l));
  PetscCall(MeshIBGetLocalIndex_Internal(mesh, l, i, j, k, &idx));
  *phi = PetscRealPart(mesh->ib.arr_phi[idx]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshIBGetBodyId_Internal(Mesh mesh, DMStagStencilLocation loc, PetscInt i, PetscInt j, PetscInt k, PetscInt *body)
{
  const MeshIBLocation *l;
  PetscInt              idx;

  PetscFunctionBegin;
  PetscCall(MeshIBFindLocation_Private(mesh, loc, &l));
  PetscCall(MeshIBGetLocalIndex_Internal(mesh, l, i, j, k, &idx));
  *body = (PetscInt)mesh->ib.body_id[idx];
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshIBGetPointCoordinate_Internal(Mesh mesh, DMStagStencilLocation loc, PetscInt i, PetscInt j, PetscInt k, PetscReal x[])
{
  const MeshIB *ib = &mesh->ib;
  PetscInt      p[3], d;

  PetscFunctionBegin;
  p[0] = i;
  p[1] = j;
  p[2] = k;
  for (d = 0; d < 3; ++d) {
    if (d >= mesh->dim) {
      x[d] = 0.;
      continue;
    }
    PetscCheck(ib->gx[d] <= p[d] && p[d] < ib->gx[d] + ib->gn[d], PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") is outside the local ghosted range", i, j, k);
    x[d] = PetscRealPart(ib->arr_coord[d][p[d]][MeshIBIsFaceInDir_Private(loc, d) ? ib->slot_coord_prev : ib->slot_coord_elem]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshIBGetForcingPoint_Internal(Mesh mesh, DMStagStencilLocation loc, PetscInt i, PetscInt j, PetscInt k, const MeshIBForcingPoint **fp)
{
  const MeshIBLocation *l;
  PetscInt              idx, r;

  PetscFunctionBegin;
  PetscCall(MeshIBFindLocation_Private(mesh, loc, &l));
  PetscCall(MeshIBGetLocalIndex_Internal(mesh, l, i, j, k, &idx));
  PetscCall(PetscHMapIGet(l->forcing_map, idx, &r));
  *fp = r >= 0 ? &l->forcing[r] : NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshIBEvaluateBodyVelocity_Internal(Mesh mesh, PetscInt body, PetscReal t, const PetscReal x[], PetscBool dot, PetscScalar u[])
{
  PetscFunctionBegin;
  PetscCheck(body >= 0 && body < mesh->ib.nbodies, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Body %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", body, mesh->ib.nbodies);
  PetscCall(IBBodyEvaluateVelocity_Internal(mesh->ib.bodies[body], mesh->dim, t, x, dot, u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshIBDestroy_Internal(Mesh mesh)
{
  MeshIB  *ib = &mesh->ib;
  PetscInt n;

  PetscFunctionBegin;
  if (ib->arr_phi) PetscCall(VecRestoreArrayRead(ib->phi_local, &ib->arr_phi));
  if (ib->arr_coord[0]) PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->dm, &ib->arr_coord[0], &ib->arr_coord[1], &ib->arr_coord[2]));
  for (n = 0; n < ib->nloc; ++n) {
    PetscCall(PetscFree(ib->locs[n].forcing));
    PetscCall(PetscHMapIDestroy(&ib->locs[n].forcing_map));
  }
  PetscCall(PetscFree(ib->body_id));
  PetscCall(VecDestroy(&ib->phi_global));
  PetscCall(VecDestroy(&ib->phi_local));
  PetscCall(DMDestroy(&ib->dm_phi));
  for (n = 0; n < ib->nbodies; ++n) PetscCall(IBBodyDestroy(&ib->bodies[n]));
  PetscCall(PetscFree(ib->bodies));
  PetscCall(PetscMemzero(ib, sizeof(MeshIB)));
  PetscFunctionReturn(PETSC_SUCCESS);
}
