#include <fluca/private/flucafdimpl.h>
#include <fluca/private/meshimpl.h>

#define FLUCAFD_IB_NORMAL_TOL 1e-12

/* Lagrange weights at x_f for one cut: a[0] for the crossing, a[m] for the fluid node m steps away from the cut side.
   *nnodes counts the crossing plus the usable fluid nodes; it is 0 when the first fluid node is solid or off the grid. */
static PetscErrorCode GetCutWeights_Private(FlucaFD fd, PetscInt order, const PetscInt p[], const MeshIBCut *cut, PetscInt *nnodes, PetscReal a[])
{
  PetscInt  sigma, m, b, c, q[3];
  PetscReal s[3], xf[3], xq[3], phi;
  PetscBool on;

  PetscFunctionBegin;
  *nnodes = 0;
  sigma   = cut->side ? 1 : -1;
  s[0]    = cut->x_ib[cut->dir];
  PetscCall(MeshIBGetPointCoordinate_Internal(fd->mesh, fd->output_loc, p[0], p[1], p[2], xf));
  for (m = 1; m <= order; ++m) {
    q[0] = p[0];
    q[1] = p[1];
    q[2] = p[2];
    q[cut->dir] -= m * sigma;
    PetscCall(MeshIBIsOnGrid_Internal(fd->mesh, fd->output_loc, cut->dir, q[cut->dir], &on));
    if (!on) break;
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

/* Usable cuts of a forcing point, their blend weights n_d^2 / sum n_d^2, and their interpolation weights */
static PetscErrorCode GetForcingRow_Private(FlucaFD fd, const PetscInt p[], const MeshIBForcingPoint *fp, PetscInt *nuse, PetscInt use[], PetscReal w[], PetscInt nnodes[], PetscReal a[][3])
{
  FlucaFD_IBForcing *ibf = (FlucaFD_IBForcing *)fd->data;
  PetscInt           n, nn, u;
  PetscReal          sum = 0.;

  PetscFunctionBegin;
  *nuse = 0;
  for (n = 0; n < fp->ncut; ++n) {
    PetscCall(GetCutWeights_Private(fd, ibf->order, p, &fp->cuts[n], &nn, a[*nuse]));
    if (!nn) continue;
    use[*nuse]    = n;
    nnodes[*nuse] = nn;
    w[*nuse]      = fp->normal[fp->cuts[n].dir] * fp->normal[fp->cuts[n].dir];
    sum += w[*nuse];
    ++(*nuse);
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
  PetscMPIInt               rank, size, first, root;
  PetscInt                  s[3], e[3], m[3], ne[3], p[3], d, nb, nuse, use[2 * FLUCAFD_MAX_DIM], nnodes[2 * FLUCAFD_MAX_DIM];
  PetscReal                 w[2 * FLUCAFD_MAX_DIM], a[2 * FLUCAFD_MAX_DIM][3], x[3] = {0., 0., 0.};
  PetscBool                 face, bad = PETSC_FALSE;
  MeshIBPointType           type;
  const MeshIBForcingPoint *fp;

  PetscFunctionBegin;
  PetscCheck(fd->input_loc == fd->output_loc && fd->input_c == fd->output_c, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_WRONGSTATE, "FLUCAFDIBFORCING requires the same input and output location and component");
  PetscCheck(ibf->order == 1 || ibf->order == 2, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Interpolation order must be 1 or 2, got %" PetscInt_FMT, ibf->order);
  PetscCheck(fd->stencil_width >= ibf->order, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_INCOMP, "Interpolation order %" PetscInt_FMT " needs a DM stencil width of at least %" PetscInt_FMT ", got %" PetscInt_FMT, ibf->order, ibf->order, fd->stencil_width);
  PetscCheck(ibf->vel_comp >= 0 && ibf->vel_comp < fd->dim, PetscObjectComm((PetscObject)fd), PETSC_ERR_ARG_OUTOFRANGE, "Velocity component %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", ibf->vel_comp, fd->dim);
  PetscCall(MeshGetNumBodies(fd->mesh, &nb));
  if (!nb) PetscFunctionReturn(PETSC_SUCCESS);

  /* Fail early, not in the solve, when a forcing point has no usable grid line; the lowest offending rank reports its first point */
  PetscCall(PetscObjectGetComm((PetscObject)fd, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
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
  for (p[2] = s[2]; p[2] < e[2] && !bad; ++p[2])
    for (p[1] = s[1]; p[1] < e[1] && !bad; ++p[1])
      for (p[0] = s[0]; p[0] < e[0] && !bad; ++p[0]) {
        PetscCall(MeshGetIBPointType(fd->mesh, fd->output_loc, p[0], p[1], p[2], &type));
        if (type != MESH_IB_FORCING) continue;
        PetscCall(MeshIBGetForcingPoint_Internal(fd->mesh, fd->output_loc, p[0], p[1], p[2], &fp));
        PetscCall(GetForcingRow_Private(fd, p, fp, &nuse, use, w, nnodes, a));
        if (nuse) continue;
        PetscCall(MeshIBGetPointCoordinate_Internal(fd->mesh, fd->output_loc, p[0], p[1], p[2], x));
        bad = PETSC_TRUE;
      }
  first = bad ? rank : size;
  PetscCallMPI(MPIU_Allreduce(&first, &root, 1, MPI_INT, MPI_MIN, comm));
  if (root < size) {
    PetscCallMPI(MPI_Bcast(x, 3, MPIU_REAL, root, comm));
    SETERRQ(comm, PETSC_ERR_USER, "Grid too coarse at the immersed boundary near (%g, %g, %g): no grid line with a fluid neighbour", (double)x[0], (double)x[1], (double)x[2]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDGetStencilRaw_IBForcing(FlucaFD fd, PetscInt i, PetscInt j, PetscInt k, PetscBool pattern, PetscInt *npoints, FlucaFDStencilPoint points[])
{
  FlucaFD_IBForcing        *ibf = (FlucaFD_IBForcing *)fd->data;
  PetscInt                  p[3], u, mm, sigma, nuse, use[2 * FLUCAFD_MAX_DIM], nnodes[2 * FLUCAFD_MAX_DIM];
  PetscReal                 w[2 * FLUCAFD_MAX_DIM], a[2 * FLUCAFD_MAX_DIM][3];
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
  PetscCheck(fp, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No forcing record at (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT "): FLUCAFDIBFORCING rows exist only at owned points, so use it as the outermost operator, a summand, or a FLUCAFDIBSELECT operand, not as an inner operand", i, j, k);
  PetscCall(GetForcingRow_Private(fd, p, fp, &nuse, use, w, nnodes, a));
  PetscCheck(nuse > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Forcing point (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") has no usable grid line", i, j, k);
  for (u = 0; u < nuse; ++u) {
    cut   = &fp->cuts[use[u]];
    sigma = cut->side ? 1 : -1;

    pt.type          = FLUCAFD_STENCIL_IB;
    pt.i             = i;
    pt.j             = j;
    pt.k             = k;
    pt.c             = ibf->vel_comp;
    pt.boundary_face = 2 * cut->dir + cut->side;
    pt.v             = -w[u] * a[u][0];
    PetscCall(FlucaFDAddStencilPoint_Internal(&pt, npoints, points));

    for (mm = 1; mm < nnodes[u]; ++mm) {
      pt.type          = FLUCAFD_STENCIL_GRID;
      pt.i             = cut->dir == 0 ? i - mm * sigma : i;
      pt.j             = cut->dir == 1 ? j - mm * sigma : j;
      pt.k             = cut->dir == 2 ? k - mm * sigma : k;
      pt.c             = fd->output_c;
      pt.boundary_face = 0;
      pt.v             = -w[u] * a[u][mm];
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
