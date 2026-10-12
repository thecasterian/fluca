#include <fluca/private/flucafdimpl.h>
#include <fluca/private/meshimpl.h>

#define FLUCAFD_IB_NORMAL_TOL 1e-12

/* Face location normal to each direction, where the boundary node of an ELEMENT forcing row lives */
static const DMStagStencilLocation wall_loc[] = {DMSTAG_LEFT, DMSTAG_DOWN, DMSTAG_BACK};

/* Lagrange weights at x_f for one cut: a[0] for the crossing, a[m] for the m-th node away from the cut side.
   The nodes are the fluid points f - s and f - 2s on the grid; when f - s is off the grid at a domain face that has a
   Dirichlet BC on fd (ELEMENT output only), that face is the single node and *wall is set.
   *nnodes counts the crossing plus the usable nodes; it is 0 when the cut has no usable node. */
static PetscErrorCode GetCutWeights_Private(FlucaFD fd, PetscInt order, const PetscInt p[], const MeshIBCut *cut, PetscInt *nnodes, PetscReal a[], PetscBool *wall)
{
  PetscInt  sigma, m, b, c, q[3], wf;
  PetscReal s[3], xf[3], xq[3], phi;
  PetscBool on;

  PetscFunctionBegin;
  *nnodes = 0;
  *wall   = PETSC_FALSE;
  sigma   = cut->side ? 1 : -1;
  s[0]    = cut->x_ib[cut->dir];
  PetscCall(MeshIBGetPointCoordinate_Internal(fd->mesh, fd->output_loc, p[0], p[1], p[2], xf));
  for (m = 1; m <= order; ++m) {
    q[0] = p[0];
    q[1] = p[1];
    q[2] = p[2];
    q[cut->dir] -= m * sigma;
    PetscCall(MeshIBIsOnGrid_Internal(fd->mesh, fd->output_loc, cut->dir, q[cut->dir], &on));
    if (!on) {
      wf = 2 * cut->dir + (1 - cut->side);
      if (m == 1 && fd->output_loc == DMSTAG_ELEMENT && fd->bcs[fd->output_c][wf].type == FLUCAFD_BC_DIRICHLET) {
        q[cut->dir] = cut->side ? 0 : fd->mesh->ib.N[cut->dir];
        PetscCall(MeshIBGetPointCoordinate_Internal(fd->mesh, wall_loc[cut->dir], q[0], q[1], q[2], xq));
        s[1]  = xq[cut->dir];
        *wall = PETSC_TRUE;
        m     = 2; /* with the break, *nnodes = 2: the crossing node and the wall node */
      }
      break;
    }
    PetscCall(MeshIBGetLevelSet_Internal(fd->mesh, fd->output_loc, q[0], q[1], q[2], &phi));
    if (phi < 0.) break;
    PetscCall(MeshIBGetPointCoordinate_Internal(fd->mesh, fd->output_loc, q[0], q[1], q[2], xq));
    s[m] = xq[cut->dir];
  }
  if (m == 1) PetscFunctionReturn(PETSC_SUCCESS);
  *nnodes = m;
  for (b = 0; b < m; ++b) {
    a[b] = 1.;
    for (c = 0; c < m; ++c)
      if (c != b) a[b] *= (xf[cut->dir] - s[c]) / (s[b] - s[c]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Usable cuts of a forcing point, their blend weights n_d^2 / sum n_d^2, their interpolation weights, and whether each
   uses a domain wall node. With no usable cut the row is zero order (*zero): every cut, a[n][0] = 1, no other node. */
static PetscErrorCode GetForcingRow_Private(FlucaFD fd, const PetscInt p[], const MeshIBForcingPoint *fp, PetscInt *nuse, PetscInt use[], PetscReal w[], PetscInt nnodes[], PetscReal a[][3], PetscBool wall[], PetscBool *zero)
{
  FlucaFD_IBForcing *ibf = (FlucaFD_IBForcing *)fd->data;
  PetscInt           n, nn, u;
  PetscReal          sum = 0.;
  PetscBool          wl;

  PetscFunctionBegin;
  *nuse = 0;
  *zero = PETSC_FALSE;
  for (n = 0; n < fp->ncut; ++n) {
    PetscCall(GetCutWeights_Private(fd, ibf->order, p, &fp->cuts[n], &nn, a[*nuse], &wl));
    if (!nn) continue;
    use[*nuse]    = n;
    nnodes[*nuse] = nn;
    wall[*nuse]   = wl;
    w[*nuse]      = fp->normal[fp->cuts[n].dir] * fp->normal[fp->cuts[n].dir];
    sum += w[*nuse];
    ++(*nuse);
  }
  if (!*nuse) {
    *zero = PETSC_TRUE;
    for (n = 0; n < fp->ncut; ++n) {
      use[n]    = n;
      nnodes[n] = 1;
      wall[n]   = PETSC_FALSE;
      a[n][0]   = 1.;
      w[n]      = fp->normal[fp->cuts[n].dir] * fp->normal[fp->cuts[n].dir];
      sum += w[n];
    }
    *nuse = fp->ncut;
  }
  for (u = 0; u < *nuse; ++u) w[u] = sum < FLUCAFD_IB_NORMAL_TOL ? 1. / *nuse : w[u] / sum;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDSetFromOptions_IBForcing(FlucaFD fd, PetscOptionItems PetscOptionsObject)
{
  FlucaFD_IBForcing *ibf = (FlucaFD_IBForcing *)fd->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "FlucaFDIBForcing Options");
  PetscCall(PetscOptionsInt("-flucafd_ib_interpolation_order", "Interpolation order along grid lines (1 or 2)", "FlucaFDIBForcingSetInterpolationOrder", ibf->order, &ibf->order, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDSetUp_IBForcing(FlucaFD fd)
{
  FlucaFD_IBForcing        *ibf = (FlucaFD_IBForcing *)fd->data;
  MPI_Comm                  comm;
  PetscInt                  s[3], e[3], m[3], ne[3], p[3], d, u, nb, nuse, use[2 * FLUCAFD_MAX_DIM], nnodes[2 * FLUCAFD_MAX_DIM], cnt[2] = {0, 0};
  PetscReal                 w[2 * FLUCAFD_MAX_DIM], a[2 * FLUCAFD_MAX_DIM][3];
  PetscBool                 face, wall[2 * FLUCAFD_MAX_DIM], zero;
  MeshIBPointType           type;
  const MeshIBForcingPoint *fp;

  PetscFunctionBegin;
  PetscCheck(fd->input_loc == fd->output_loc && fd->input_c == fd->output_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "FLUCAFDIBFORCING requires the same input and output location and component");
  PetscCheck(ibf->order == 1 || ibf->order == 2, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Interpolation order must be 1 or 2, got %" PetscInt_FMT, ibf->order);
  PetscCheck(fd->stencil_width >= ibf->order, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "Interpolation order %" PetscInt_FMT " needs a DM stencil width of at least %" PetscInt_FMT ", got %" PetscInt_FMT, ibf->order, ibf->order, fd->stencil_width);
  PetscCheck(ibf->vel_comp >= 0 && ibf->vel_comp < fd->dim, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Velocity component %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", ibf->vel_comp, fd->dim);
  PetscCall(MeshGetNumBodies(fd->mesh, &nb));
  if (!nb) PetscFunctionReturn(PETSC_SUCCESS);

  /* Report how many forcing points use a fallback: cnt[0] a domain wall node, cnt[1] zero order */
  PetscCall(PetscObjectGetComm((PetscObject)fd, &comm));
  PetscCall(DMStagGetCorners(fd->dm, &s[0], &s[1], &s[2], &m[0], &m[1], &m[2], &ne[0], &ne[1], &ne[2]));
  for (d = 0; d < 3; ++d) {
    if (d >= fd->dim) {
      s[d] = 0;
      e[d] = 1;
      continue;
    }
    PetscCall(FlucaFDUseFaceCoordinate_Internal(fd->output_loc, d, &face));
    e[d] = s[d] + m[d] + (face ? ne[d] : 0);
  }
  for (p[2] = s[2]; p[2] < e[2]; ++p[2])
    for (p[1] = s[1]; p[1] < e[1]; ++p[1])
      for (p[0] = s[0]; p[0] < e[0]; ++p[0]) {
        PetscCall(MeshGetIBPointType(fd->mesh, fd->output_loc, p[0], p[1], p[2], &type));
        if (type != MESH_IB_FORCING) continue;
        PetscCall(MeshIBGetForcingPoint_Internal(fd->mesh, fd->output_loc, p[0], p[1], p[2], &fp));
        PetscCall(GetForcingRow_Private(fd, p, fp, &nuse, use, w, nnodes, a, wall, &zero));
        if (zero) ++cnt[1];
        for (u = 0; u < nuse; ++u)
          if (wall[u]) {
            ++cnt[0];
            break;
          }
      }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, cnt, 2, MPIU_INT, MPI_SUM, comm));
  PetscCall(PetscInfo(fd, "%" PetscInt_FMT " forcing points use a domain wall node, %" PetscInt_FMT " fall back to zero order\n", cnt[0], cnt[1]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetStencilRaw_IBForcing(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscInt *npoints, FlucaFDStencilPoint points[])
{
  FlucaFD_IBForcing        *ibf = (FlucaFD_IBForcing *)fd->data;
  PetscInt                  p[3], u, mm, sigma, nuse, use[2 * FLUCAFD_MAX_DIM], nnodes[2 * FLUCAFD_MAX_DIM];
  PetscReal                 w[2 * FLUCAFD_MAX_DIM], a[2 * FLUCAFD_MAX_DIM][3];
  PetscBool                 wall[2 * FLUCAFD_MAX_DIM], zero;
  MeshIBPointType           type;
  const MeshIBForcingPoint *fp;
  const MeshIBCut          *cut;
  FlucaFDStencilPoint       pt = {0};

  PetscFunctionBegin;
  *npoints = 0;
  PetscCall(MeshGetIBPointType(fd->mesh, fd->output_loc, i, j, k, &type)); /* FLUID everywhere when the Mesh has no bodies */
  if (type == MESH_IB_FLUID) PetscFunctionReturn(PETSC_SUCCESS);

  pt.type = FLUCAFD_STENCIL_GRID;
  pt.loc  = fd->output_loc;
  pt.i    = i;
  pt.j    = j;
  pt.k    = k;
  pt.c    = fd->output_c;
  pt.v    = 1.;
  PetscCall(FlucaFDAddStencilPoint_Internal(&pt, npoints, points));

  pt.type = FLUCAFD_STENCIL_IB;
  pt.c    = ibf->vel_comp;
  if (type == MESH_IB_SOLID) {
    pt.boundary_face = -1;
    pt.v             = -1.;
    PetscCall(FlucaFDAddStencilPoint_Internal(&pt, npoints, points));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  p[0] = i;
  p[1] = j;
  p[2] = k;
  PetscCall(MeshIBGetForcingPoint_Internal(fd->mesh, fd->output_loc, i, j, k, &fp));
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No forcing record at (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "): FLUCAFDIBFORCING rows exist only at owned points, so use it as the outermost operator or a summand, not as an inner operand", i, j, k);
  PetscCall(GetForcingRow_Private(fd, p, fp, &nuse, use, w, nnodes, a, wall, &zero));
  PetscCheck(nuse > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forcing point (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") has no usable grid line", i, j, k);
  for (u = 0; u < nuse; ++u) {
    cut   = &fp->cuts[use[u]];
    sigma = cut->side ? 1 : -1;

    pt.type          = FLUCAFD_STENCIL_IB;
    pt.loc           = fd->output_loc;
    pt.i             = i;
    pt.j             = j;
    pt.k             = k;
    pt.c             = ibf->vel_comp;
    pt.boundary_face = 2 * cut->dir + cut->side;
    pt.v             = -w[u] * a[u][0];
    PetscCall(FlucaFDAddStencilPoint_Internal(&pt, npoints, points));

    for (mm = 1; mm < nnodes[u]; ++mm) {
      if (wall[u]) {
        pt.type          = FLUCAFD_STENCIL_BOUNDARY;
        pt.loc           = wall_loc[cut->dir];
        pt.i             = cut->dir == 0 ? (cut->side ? 0 : fd->mesh->ib.N[0]) : i;
        pt.j             = cut->dir == 1 ? (cut->side ? 0 : fd->mesh->ib.N[1]) : j;
        pt.k             = cut->dir == 2 ? (cut->side ? 0 : fd->mesh->ib.N[2]) : k;
        pt.c             = fd->output_c;
        pt.boundary_face = 2 * cut->dir + (1 - cut->side);
      } else {
        pt.type          = FLUCAFD_STENCIL_GRID;
        pt.loc           = fd->output_loc;
        pt.i             = cut->dir == 0 ? i - mm * sigma : i;
        pt.j             = cut->dir == 1 ? j - mm * sigma : j;
        pt.k             = cut->dir == 2 ? k - mm * sigma : k;
        pt.c             = fd->output_c;
        pt.boundary_face = 0;
      }
      pt.v = -w[u] * a[u][mm];
      PetscCall(FlucaFDAddStencilPoint_Internal(&pt, npoints, points));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDView_IBForcing(FlucaFD fd, PetscViewer viewer)
{
  FlucaFD_IBForcing *ibf = (FlucaFD_IBForcing *)fd->data;
  PetscBool          isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Interpolation order: %" PetscInt_FMT "\n", ibf->order));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Velocity component: %" PetscInt_FMT "\n", ibf->vel_comp));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDDestroy_IBForcing(FlucaFD fd)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(fd->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDCreate_IBForcing(FlucaFD fd)
{
  FlucaFD_IBForcing *ibf;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ibf));
  ibf->vel_comp = 0;
  ibf->order    = 2;

  fd->data                = (void *)ibf;
  fd->ops->setfromoptions = FlucaFDSetFromOptions_IBForcing;
  fd->ops->setup          = FlucaFDSetUp_IBForcing;
  fd->ops->getstencilraw  = FlucaFDGetStencilRaw_IBForcing;
  fd->ops->destroy        = FlucaFDDestroy_IBForcing;
  fd->ops->view           = FlucaFDView_IBForcing;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDIBForcingCreate(Mesh mesh, DMStagStencilLocation loc, PetscInt c, PetscInt vel_comp, FlucaFD *fd)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mesh, MESH_CLASSID, 1);
  PetscAssertPointer(fd, 5);
  PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));
  PetscCall(FlucaFDCreate(comm, fd));
  PetscCall(FlucaFDSetType(*fd, FLUCAFDIBFORCING));
  PetscCall(FlucaFDSetMesh(*fd, mesh));
  PetscCall(FlucaFDSetInputLocation(*fd, loc, c));
  PetscCall(FlucaFDSetOutputLocation(*fd, loc, c));
  PetscCall(FlucaFDIBForcingSetVelocityComponent(*fd, vel_comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDIBForcingSetInterpolationOrder(FlucaFD fd, PetscInt order)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDIBFORCING);
  PetscValidLogicalCollectiveInt(fd, order, 2);
  PetscCheck(!fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the interpolation order after FlucaFDSetUp()");
  ((FlucaFD_IBForcing *)fd->data)->order = order;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDIBForcingSetVelocityComponent(FlucaFD fd, PetscInt vel_comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(fd, FLUCAFD_CLASSID, 1, FLUCAFDIBFORCING);
  PetscValidLogicalCollectiveInt(fd, vel_comp, 2);
  PetscCheck(!fd->setupcalled, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the velocity component after FlucaFDSetUp()");
  ((FlucaFD_IBForcing *)fd->data)->vel_comp = vel_comp;
  PetscFunctionReturn(PETSC_SUCCESS);
}
