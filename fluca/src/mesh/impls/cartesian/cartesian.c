#include <fluca/private/meshimpl.h>
#include <petscdmstag.h>

static PetscErrorCode MeshSetUp_Cartesian(Mesh mesh)
{
  DM        cdm;
  PetscInt  sw;
  PetscBool isproduct;

  PetscFunctionBegin;
  PetscCall(DMStagGetStencilWidth(mesh->dm, &sw));
  PetscCheck(sw >= 1, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONG, "MeshCartesian requires a DM stencil width of at least 1, got %" PetscInt_FMT, sw);
  PetscCall(DMGetCoordinateDM(mesh->dm, &cdm));
  PetscCall(PetscObjectTypeCompare((PetscObject)cdm, DMPRODUCT, &isproduct));
  PetscCheck(isproduct, PetscObjectComm((PetscObject)mesh), PETSC_ERR_ARG_WRONG, "MeshCartesian requires product coordinates; call DMStagSetUniformCoordinatesProduct() on the DM first");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Global coordinate range [lo, hi] of the vertices in each direction */
static PetscErrorCode MeshCartesianGetDomain_Private(Mesh mesh, PetscReal lo[], PetscReal hi[])
{
  const PetscScalar **arr[3] = {NULL, NULL, NULL};
  PetscInt            x[3], m[3], iprev, d;
  PetscBool           first[3], last[3];
  PetscReal           loc_lo[3], loc_hi[3];

  PetscFunctionBegin;
  PetscCall(DMStagGetCorners(mesh->dm, &x[0], &x[1], &x[2], &m[0], &m[1], &m[2], NULL, NULL, NULL));
  PetscCall(DMStagGetIsFirstRank(mesh->dm, &first[0], &first[1], &first[2]));
  PetscCall(DMStagGetIsLastRank(mesh->dm, &last[0], &last[1], &last[2]));
  PetscCall(DMStagGetProductCoordinateArraysRead(mesh->dm, &arr[0], &arr[1], &arr[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(mesh->dm, DMSTAG_LEFT, &iprev));
  for (d = 0; d < mesh->dim; ++d) {
    loc_lo[d] = first[d] ? PetscRealPart(arr[d][x[d]][iprev]) : PETSC_MAX_REAL;
    loc_hi[d] = last[d] ? PetscRealPart(arr[d][x[d] + m[d]][iprev]) : PETSC_MIN_REAL;
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(mesh->dm, &arr[0], &arr[1], &arr[2]));
  PetscCallMPI(MPIU_Allreduce(loc_lo, lo, (PetscMPIInt)mesh->dim, MPIU_REAL, MPIU_MIN, PetscObjectComm((PetscObject)mesh)));
  PetscCallMPI(MPIU_Allreduce(loc_hi, hi, (PetscMPIInt)mesh->dim, MPIU_REAL, MPIU_MAX, PetscObjectComm((PetscObject)mesh)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MeshView_Cartesian(Mesh mesh, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscInt       M[3], d;
  DMBoundaryType bt[3];
  PetscReal      lo[3], hi[3];

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (!isascii || !mesh->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMStagGetGlobalSizes(mesh->dm, &M[0], &M[1], &M[2]));
  PetscCall(DMStagGetBoundaryTypes(mesh->dm, &bt[0], &bt[1], &bt[2]));
  PetscCall(MeshCartesianGetDomain_Private(mesh, lo, hi));

  PetscCall(PetscViewerASCIIPushTab(viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Dimension: %" PetscInt_FMT "\n", mesh->dim));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Global sizes: %" PetscInt_FMT, M[0]));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
  for (d = 1; d < mesh->dim; ++d) PetscCall(PetscViewerASCIIPrintf(viewer, " x %" PetscInt_FMT, M[d]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Boundary types:"));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
  for (d = 0; d < mesh->dim; ++d) PetscCall(PetscViewerASCIIPrintf(viewer, " %s", DMBoundaryTypes[bt[d]]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  PetscCall(PetscViewerASCIIPrintf(viewer, "Domain: [%g, %g]", (double)lo[0], (double)hi[0]));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
  for (d = 1; d < mesh->dim; ++d) PetscCall(PetscViewerASCIIPrintf(viewer, " x [%g, %g]", (double)lo[d], (double)hi[d]));
  PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCreate_Cartesian(Mesh mesh)
{
  PetscFunctionBegin;
  mesh->data       = NULL;
  mesh->ops->setup = MeshSetUp_Cartesian;
  mesh->ops->view  = MeshView_Cartesian;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MeshCartesianCreate(DM dm, Mesh *mesh)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(dm, DM_CLASSID, 1, DMSTAG);
  PetscAssertPointer(mesh, 2);
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(MeshCreate(comm, mesh));
  PetscCall(MeshSetType(*mesh, MESHCARTESIAN));
  PetscCall(MeshSetDM(*mesh, dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
