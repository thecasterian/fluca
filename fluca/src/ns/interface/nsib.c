#include <fluca/private/nsimpl.h>
#include <fluca/private/meshimpl.h>

/* Global index of a point of the solution DM given by its ghosted-local position */
PetscErrorCode NSIBGetGlobalIndex_Internal(DM dm, const DMStagStencil *st, PetscInt *g)
{
  ISLocalToGlobalMapping ltog;
  PetscInt               dim, l;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagStencilToIndexLocal(dm, dim, 1, st, &l));
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  PetscCall(ISLocalToGlobalMappingApply(ltog, 1, &l, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Whether cell q (owned or ghost) is SOLID; a cell off a non-periodic grid is not */
PetscErrorCode NSIBIsSolidCell_Internal(Mesh mesh, PetscInt dim, const PetscInt q[], PetscBool *solid)
{
  MeshIBPointType type;
  PetscBool       on;
  PetscInt        d;

  PetscFunctionBegin;
  *solid = PETSC_FALSE;
  for (d = 0; d < dim; ++d) {
    PetscCall(MeshIBIsOnGrid_Internal(mesh, DMSTAG_ELEMENT, d, q[d], &on));
    if (!on) PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(MeshGetIBPointType(mesh, DMSTAG_ELEMENT, q[0], q[1], q[2], &type));
  *solid = (PetscBool)(type == MESH_IB_SOLID);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSortedIS_Private(MPI_Comm comm, PetscInt n, PetscInt idx[], IS *is)
{
  PetscFunctionBegin;
  PetscCall(PetscSortInt(n, idx));
  PetscCall(ISCreateGeneral(comm, n, idx, PETSC_COPY_VALUES, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Build the IB row sets and the IBFORCING operators; nothing without bodies. The Mesh geometry is stationary, so
   this runs once. */
PetscErrorCode NSSetUpIB_Internal(NS ns)
{
  MPI_Comm        comm;
  Mesh            mesh;
  DM              dm;
  MeshIBPointType type;
  DMStagStencil   st;
  NS_IBFace      *rec;
  PetscInt        nb, dim, c_vel, c_U, c_p, xs[3], xm[3], p[3], q[3], d, e, ncell, nmom = 0, nunit = 0, nface = 0;
  PetscInt       *mom, *unit, *face;
  PetscReal       phi_lo, phi_hi;
  PetscBool       on, solid_lo, solid_hi;

  PetscFunctionBegin;
  PetscCall(PhysGetMesh(ns->phys, &mesh));
  PetscCall(MeshGetNumBodies(mesh, &nb));
  if (!nb) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(PhysGetField(ns->phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(DMStagGetCorners(dm, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2], NULL, NULL, NULL));
  for (d = dim; d < 3; ++d) {
    xs[d] = 0;
    xm[d] = 1;
  }
  ncell = xm[0] * xm[1] * xm[2];
  PetscCall(PetscMalloc1(dim * ncell, &mom));
  PetscCall(PetscMalloc1((dim + 1) * ncell, &unit));
  PetscCall(PetscMalloc1(dim * ncell, &face));
  PetscCall(PetscMalloc1(dim * ncell, &ns->ibface));

  for (p[2] = xs[2]; p[2] < xs[2] + xm[2]; ++p[2])
    for (p[1] = xs[1]; p[1] < xs[1] + xm[1]; ++p[1])
      for (p[0] = xs[0]; p[0] < xs[0] + xm[0]; ++p[0]) {
        st.i   = p[0];
        st.j   = p[1];
        st.k   = p[2];
        st.loc = DMSTAG_ELEMENT;
        PetscCall(MeshGetIBPointType(mesh, DMSTAG_ELEMENT, p[0], p[1], p[2], &type));
        if (type != MESH_IB_FLUID) {
          for (d = 0; d < dim; ++d) {
            st.c = c_vel + d;
            PetscCall(NSIBGetGlobalIndex_Internal(dm, &st, &mom[nmom++]));
          }
        }
        if (type == MESH_IB_SOLID) {
          st.c = c_p;
          PetscCall(NSIBGetGlobalIndex_Internal(dm, &st, &unit[nunit++]));
        }
        /* The face normal to e on the low side of this cell; domain-boundary faces keep the domain BC */
        for (e = 0; e < dim; ++e) {
          q[0] = p[0];
          q[1] = p[1];
          q[2] = p[2];
          --q[e];
          PetscCall(MeshIBIsOnGrid_Internal(mesh, DMSTAG_ELEMENT, e, q[e], &on));
          if (!on) continue;
          PetscCall(NSIBIsSolidCell_Internal(mesh, dim, q, &solid_lo));
          solid_hi = (PetscBool)(type == MESH_IB_SOLID);
          if (!solid_lo && !solid_hi) continue;
          rec         = &ns->ibface[nface];
          rec->st.i   = p[0];
          rec->st.j   = p[1];
          rec->st.k   = p[2];
          rec->st.loc = face_loc[e];
          rec->st.c   = c_U;
          rec->e      = e;
          if (solid_lo && solid_hi) {
            PetscCall(MeshIBGetLevelSet_Internal(mesh, DMSTAG_ELEMENT, q[0], q[1], q[2], &phi_lo));
            PetscCall(MeshIBGetLevelSet_Internal(mesh, DMSTAG_ELEMENT, p[0], p[1], p[2], &phi_hi));
            if (phi_lo < phi_hi) PetscCall(MeshIBGetBodyId_Internal(mesh, DMSTAG_ELEMENT, q[0], q[1], q[2], &rec->body));
            else PetscCall(MeshIBGetBodyId_Internal(mesh, DMSTAG_ELEMENT, p[0], p[1], p[2], &rec->body));
          } else if (solid_lo) PetscCall(MeshIBGetBodyId_Internal(mesh, DMSTAG_ELEMENT, q[0], q[1], q[2], &rec->body));
          else PetscCall(MeshIBGetBodyId_Internal(mesh, DMSTAG_ELEMENT, p[0], p[1], p[2], &rec->body));
          PetscCall(MeshIBGetPointCoordinate_Internal(mesh, face_loc[e], p[0], p[1], p[2], rec->x));
          PetscCall(NSIBGetGlobalIndex_Internal(dm, &rec->st, &face[nface]));
          unit[nunit++] = face[nface];
          ++nface;
        }
      }
  ns->nibface = nface;
  PetscCall(CreateSortedIS_Private(comm, nmom, mom, &ns->is_ib_mom));
  PetscCall(CreateSortedIS_Private(comm, nface, face, &ns->is_ib_face));
  PetscCall(CreateSortedIS_Private(comm, nunit, unit, &ns->is_ib_unit));
  PetscCall(PetscFree(face));
  PetscCall(PetscFree(unit));
  PetscCall(PetscFree(mom));

  for (d = 0; d < dim; ++d) {
    PetscCall(FlucaFDIBForcingCreate(mesh, DMSTAG_ELEMENT, c_vel + d, d, &ns->fd_ibforcing[d]));
    PetscCall(NSSetVelocityBCs_Internal(ns, ns->fd_ibforcing[d], d));
    PetscCall(FlucaFDSetFromOptions(ns->fd_ibforcing[d]));
    PetscCall(FlucaFDSetUp(ns->fd_ibforcing[d]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* J: momentum rows at FORCING/SOLID cells become IBFORCING rows, IB face and solid pressure rows become unit rows.
   J must be assembled and have the nonzero pattern of DMCreateMatrix() on the solution DM. */
PetscErrorCode NSIBReplaceJacobianRows_Internal(NS ns, Mat J)
{
  DM       dm;
  PetscInt dim, d;

  PetscFunctionBegin;
  if (!ns->is_ib_mom) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(MatSetOption(J, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(MatZeroRowsIS(J, ns->is_ib_mom, 0., NULL, NULL));
  PetscCall(MatZeroRowsIS(J, ns->is_ib_unit, 1., NULL, NULL));
  for (d = 0; d < dim; ++d) PetscCall(FlucaFDGetOperator(ns->fd_ibforcing[d], dm, dm, J));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* f: the IB rows get the wall values at the end of the step, t + dt (the coupled system solves for that state):
   momentum b_ib = -IBFORCING(0), IB faces u_body . e_e, solid pressures 0 */
PetscErrorCode NSIBReplaceFunctionRows_Internal(NS ns, Vec f)
{
  PetscReal   t = ns->t + ns->dt;
  PetscScalar u[3], v;
  PetscInt    dim, d, n;
  Mesh        mesh;
  DM          dm;
  Vec         tmp;

  PetscFunctionBegin;
  if (!ns->is_ib_mom) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PhysGetMesh(ns->phys, &mesh));
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(VecISSet(f, ns->is_ib_mom, 0.));
  PetscCall(VecISSet(f, ns->is_ib_unit, 0.));
  PetscCall(DMGetGlobalVector(dm, &tmp));
  for (d = 0; d < dim; ++d) {
    PetscCall(VecZeroEntries(tmp));
    PetscCall(FlucaFDApply(ns->fd_ibforcing[d], t, dm, dm, ns->zero, tmp));
    PetscCall(VecAXPY(f, -1., tmp));
  }
  PetscCall(DMRestoreGlobalVector(dm, &tmp));
  for (n = 0; n < ns->nibface; ++n) {
    PetscCall(MeshIBEvaluateBodyVelocity_Internal(mesh, ns->ibface[n].body, t, ns->ibface[n].x, PETSC_FALSE, u));
    v = u[ns->ibface[n].e];
    PetscCall(DMStagVecSetValuesStencil(dm, f, 1, &ns->ibface[n].st, &v, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(f));
  PetscCall(VecAssemblyEnd(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The momentum pressure gradient of a FLUID cell at a domain wall is fd_grad_p's closure through p + t and p + 2t;
   if p + 2t is SOLID it would read a decoupled pressure. That needs an IB-aware gradient in the subtype, which phase 3a
   does not have, so such a thin gap between a body and a domain wall is rejected. */
PetscErrorCode NSIBCheckWallGap_Internal(NS ns)
{
  Mesh            mesh;
  DM              dm;
  MeshIBPointType type;
  PetscInt        dim, xs[3], xm[3], p[3], q[3], d, e, t, nbad = 0;
  PetscBool       on_lo, on_hi, solid;

  PetscFunctionBegin;
  if (!ns->is_ib_mom) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PhysGetMesh(ns->phys, &mesh));
  PetscCall(PhysGetSolutionDM(ns->phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetCorners(dm, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2], NULL, NULL, NULL));
  for (d = dim; d < 3; ++d) {
    xs[d] = 0;
    xm[d] = 1;
  }
  for (p[2] = xs[2]; p[2] < xs[2] + xm[2]; ++p[2])
    for (p[1] = xs[1]; p[1] < xs[1] + xm[1]; ++p[1])
      for (p[0] = xs[0]; p[0] < xs[0] + xm[0]; ++p[0]) {
        PetscCall(MeshGetIBPointType(mesh, DMSTAG_ELEMENT, p[0], p[1], p[2], &type));
        if (type != MESH_IB_FLUID) continue;
        for (e = 0; e < dim; ++e) {
          PetscCall(MeshIBIsOnGrid_Internal(mesh, DMSTAG_ELEMENT, e, p[e] - 1, &on_lo));
          PetscCall(MeshIBIsOnGrid_Internal(mesh, DMSTAG_ELEMENT, e, p[e] + 1, &on_hi));
          if (on_lo == on_hi) continue;
          t    = on_lo ? -1 : 1;
          q[0] = p[0];
          q[1] = p[1];
          q[2] = p[2];
          q[e] = p[e] + 2 * t;
          PetscCall(NSIBIsSolidCell_Internal(mesh, dim, q, &solid));
          if (solid) ++nbad;
        }
      }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &nbad, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)ns)));
  PetscCheck(!nbad, PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "An immersed body is within two cells of a domain wall at %" PetscInt_FMT " fluid cells: the wall-closure pressure gradient would read a solid cell; refine the grid or move the body", nbad);
  PetscFunctionReturn(PETSC_SUCCESS);
}
