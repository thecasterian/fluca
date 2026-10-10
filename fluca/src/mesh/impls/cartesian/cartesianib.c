#include <fluca/private/meshimpl.h>
#include <petscdmstag.h>

/* Sample every body on the ghosted points of l->loc; keep the minimum and its body id */
static PetscErrorCode SampleLocation_Private(Mesh mesh, const MeshIBLocation *l, PetscScalar arr[])
{
  MeshIB    *ib       = &mesh->ib;
  PetscReal *coord[3] = {NULL, NULL, NULL};
  PetscInt   n[3]     = {1, 1, 1};
  PetscReal *phi;
  PetscInt   d, b, i, j, k, slot, idx;
  PetscReal  p;

  PetscFunctionBegin;
  for (d = 0; d < mesh->dim; ++d) {
    n[d] = ib->gn[d];
    slot = MeshIBIsFaceInDir_Private(l->loc, d) ? ib->slot_coord_prev : ib->slot_coord_elem;
    PetscCall(PetscMalloc1(n[d], &coord[d]));
    for (i = 0; i < n[d]; ++i) coord[d][i] = PetscRealPart(ib->arr_coord[d][ib->gx[d] + i][slot]);
  }
  PetscCall(PetscMalloc1(n[0] * n[1] * n[2], &phi));
  for (b = 0; b < ib->nbodies; ++b) {
    PetscCall(IBBodySample_Internal(ib->bodies[b], mesh->dim, n, (const PetscReal *const *)coord, phi));
    for (k = 0; k < n[2]; ++k)
      for (j = 0; j < n[1]; ++j)
        for (i = 0; i < n[0]; ++i) {
          PetscCall(MeshIBGetLocalIndex_Internal(mesh, l, ib->gx[0] + i, ib->gx[1] + j, ib->gx[2] + k, &idx));
          p = phi[(k * n[1] + j) * n[0] + i];
          if (b == 0 || p < PetscRealPart(arr[idx])) {
            arr[idx]         = p;
            ib->body_id[idx] = (int8_t)b;
          }
        }
  }
  PetscCall(PetscFree(phi));
  for (d = 0; d < 3; ++d) PetscCall(PetscFree(coord[d]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Unit grad(phi) at p by central (one-sided at the domain edge) differences; falls back to the cut directions */
static PetscErrorCode ComputeNormal_Private(Mesh mesh, const MeshIBLocation *l, const PetscInt p[], MeshIBForcingPoint *fp)
{
  const MeshIB *ib = &mesh->ib;
  PetscInt      d, s, q[3], idx;
  PetscReal     phi[2], x[2][3], norm = 0.;
  PetscBool     on;

  PetscFunctionBegin;
  for (d = 0; d < 3; ++d) fp->normal[d] = 0.;
  for (d = 0; d < mesh->dim; ++d) {
    for (s = 0; s < 2; ++s) {
      q[0] = p[0];
      q[1] = p[1];
      q[2] = p[2];
      q[d] += s ? 1 : -1;
      PetscCall(MeshIBIsOnGrid_Internal(mesh, l->loc, d, q[d], &on));
      if (!on) q[d] = p[d];
      PetscCall(MeshIBGetLocalIndex_Internal(mesh, l, q[0], q[1], q[2], &idx));
      phi[s] = PetscRealPart(ib->arr_phi[idx]);
      PetscCall(MeshIBGetPointCoordinate_Internal(mesh, l->loc, q[0], q[1], q[2], x[s]));
    }
    if (x[1][d] > x[0][d]) fp->normal[d] = (phi[1] - phi[0]) / (x[1][d] - x[0][d]);
    norm += fp->normal[d] * fp->normal[d];
  }
  norm = PetscSqrtReal(norm);
  if (norm < 1e-12) {
    /* A cut on side s means fluid lies the other way */
    for (s = 0; s < fp->ncut; ++s) fp->normal[fp->cuts[s].dir] += fp->cuts[s].side ? -1. : 1.;
    norm = 0.;
    for (d = 0; d < mesh->dim; ++d) norm += fp->normal[d] * fp->normal[d];
    norm = PetscSqrtReal(norm);
  }
  if (norm > 0.)
    for (d = 0; d < mesh->dim; ++d) fp->normal[d] /= norm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildForcingRecords_Private(Mesh mesh, MeshIBLocation *l)
{
  const MeshIB       *ib = &mesh->ib;
  PetscInt            s[3], e[3], m[3], ne[3], p[3], q[3], d, side, idx, nidx;
  PetscReal           phi_f, phi_n, xf[3], xn[3];
  PetscBool           on;
  MeshIBPointType     type;
  MeshIBForcingPoint  fp;
  MeshIBForcingPoint *grown;

  PetscFunctionBegin;
  PetscCall(PetscHMapICreate(&l->forcing_map));
  PetscCall(DMStagGetCorners(mesh->dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2], &ne[0], &ne[1], &ne[2]));
  for (d = 0; d < 3; ++d) {
    if (d >= mesh->dim) {
      s[d] = 0;
      e[d] = 1;
    } else e[d] = s[d] + m[d] + (MeshIBIsFaceInDir_Private(l->loc, d) ? ne[d] : 0);
  }
  for (p[2] = s[2]; p[2] < e[2]; ++p[2])
    for (p[1] = s[1]; p[1] < e[1]; ++p[1])
      for (p[0] = s[0]; p[0] < e[0]; ++p[0]) {
        PetscCall(MeshIBClassify_Internal(mesh, l, p[0], p[1], p[2], &type));
        if (type != MESH_IB_FORCING) continue;
        PetscCall(PetscMemzero(&fp, sizeof(fp)));
        PetscCall(MeshIBGetLocalIndex_Internal(mesh, l, p[0], p[1], p[2], &idx));
        phi_f = PetscRealPart(ib->arr_phi[idx]);
        PetscCall(MeshIBGetPointCoordinate_Internal(mesh, l->loc, p[0], p[1], p[2], xf));
        for (d = 0; d < mesh->dim; ++d) {
          for (side = 0; side < 2; ++side) {
            q[0] = p[0];
            q[1] = p[1];
            q[2] = p[2];
            q[d] += side ? 1 : -1;
            PetscCall(MeshIBIsOnGrid_Internal(mesh, l->loc, d, q[d], &on));
            if (!on) continue;
            PetscCall(MeshIBGetLocalIndex_Internal(mesh, l, q[0], q[1], q[2], &nidx));
            phi_n = PetscRealPart(ib->arr_phi[nidx]);
            if (phi_n >= 0.) continue;
            PetscCall(MeshIBGetPointCoordinate_Internal(mesh, l->loc, q[0], q[1], q[2], xn));
            fp.cuts[fp.ncut].dir   = d;
            fp.cuts[fp.ncut].side  = side;
            fp.cuts[fp.ncut].theta = phi_f / (phi_f - phi_n);
            fp.cuts[fp.ncut].body  = (PetscInt)ib->body_id[nidx];
            PetscCall(PetscArraycpy(fp.cuts[fp.ncut].x_ib, xf, 3));
            fp.cuts[fp.ncut].x_ib[d] = xf[d] + fp.cuts[fp.ncut].theta * (xn[d] - xf[d]);
            ++fp.ncut;
          }
        }
        PetscCall(ComputeNormal_Private(mesh, l, p, &fp));
        if (l->nforcing == l->maxforcing) {
          l->maxforcing = l->maxforcing ? 2 * l->maxforcing : 64;
          PetscCall(PetscMalloc1(l->maxforcing, &grown));
          if (l->nforcing) PetscCall(PetscArraycpy(grown, l->forcing, l->nforcing));
          PetscCall(PetscFree(l->forcing));
          l->forcing = grown;
        }
        PetscCall(PetscHMapISet(l->forcing_map, idx, l->nforcing));
        l->forcing[l->nforcing++] = fp;
      }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshSetUpIB_Cartesian(Mesh mesh)
{
  MeshIB                     *ib = &mesh->ib;
  PetscInt                    d, n, nlocal;
  DMBoundaryType              bt[3];
  PetscScalar                *arr;
  const DMStagStencilLocation faces[3] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

  PetscFunctionBegin;
  for (n = 0; n < ib->nbodies; ++n) PetscCall(IBBodySetUp(ib->bodies[n]));

  /* Locations: the element and every face location, whatever dofs the mesh DM has */
  switch (mesh->dim) {
  case 1:
    PetscCall(DMStagCreateCompatibleDMStag(mesh->dm, 1, 1, 0, 0, &ib->dm_phi));
    break;
  case 2:
    PetscCall(DMStagCreateCompatibleDMStag(mesh->dm, 0, 1, 1, 0, &ib->dm_phi));
    break;
  case 3:
    PetscCall(DMStagCreateCompatibleDMStag(mesh->dm, 0, 0, 1, 1, &ib->dm_phi));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)mesh), PETSC_ERR_SUP, "Unsupported dimension %" PetscInt_FMT, mesh->dim);
  }
  ib->nloc        = 0;
  ib->locs[0].loc = DMSTAG_ELEMENT;
  ++ib->nloc;
  for (d = 0; d < mesh->dim; ++d) ib->locs[ib->nloc++].loc = faces[d];
  for (n = 0; n < ib->nloc; ++n) PetscCall(DMStagGetLocationSlot(ib->dm_phi, ib->locs[n].loc, 0, &ib->locs[n].slot));

  /* Layout and coordinates */
  PetscCall(DMStagGetGhostCorners(ib->dm_phi, &ib->gx[0], &ib->gx[1], &ib->gx[2], &ib->gn[0], &ib->gn[1], &ib->gn[2]));
  PetscCall(DMStagGetEntriesPerElement(ib->dm_phi, &ib->epe));
  PetscCall(DMStagGetGlobalSizes(mesh->dm, &ib->N[0], &ib->N[1], &ib->N[2]));
  PetscCall(DMStagGetBoundaryTypes(mesh->dm, &bt[0], &bt[1], &bt[2]));
  for (d = 0; d < 3; ++d) {
    if (d >= mesh->dim) {
      ib->gx[d]       = 0;
      ib->gn[d]       = 1;
      ib->N[d]        = 1;
      ib->periodic[d] = PETSC_FALSE;
    } else ib->periodic[d] = bt[d] == DM_BOUNDARY_PERIODIC;
  }
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, &ib->arr_coord[0], &ib->arr_coord[1], &ib->arr_coord[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &ib->slot_coord_prev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_ELEMENT, &ib->slot_coord_elem));

  /* Sample phi on every ghosted entry */
  PetscCall(DMCreateLocalVector(ib->dm_phi, &ib->phi_local));
  PetscCall(VecGetLocalSize(ib->phi_local, &nlocal));
  PetscCall(PetscMalloc1(nlocal, &ib->body_id));
  for (n = 0; n < nlocal; ++n) ib->body_id[n] = -1;
  PetscCall(VecSet(ib->phi_local, 1.));
  PetscCall(VecGetArray(ib->phi_local, &arr));
  for (n = 0; n < ib->nloc; ++n) PetscCall(SampleLocation_Private(mesh, &ib->locs[n], arr));
  PetscCall(VecRestoreArray(ib->phi_local, &arr));
  PetscCall(DMCreateGlobalVector(ib->dm_phi, &ib->phi_global));
  PetscCall(PetscObjectSetName((PetscObject)ib->phi_global, "LevelSet"));
  PetscCall(DMLocalToGlobal(ib->dm_phi, ib->phi_local, INSERT_VALUES, ib->phi_global));
  PetscCall(VecGetArrayRead(ib->phi_local, &ib->arr_phi));

  /* Forcing records for owned points */
  for (n = 0; n < ib->nloc; ++n) PetscCall(BuildForcingRecords_Private(mesh, &ib->locs[n]));
  for (n = 0; n < ib->nbodies; ++n) PetscCall(IBBodyReleaseGeometry_Internal(ib->bodies[n]));
  PetscFunctionReturn(PETSC_SUCCESS);
}
