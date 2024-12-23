#include <fluca/private/flucaviewer_cgns.h>
#include <fluca/private/sol_fsm.h>
#include <pcgnslib.h>
#include <petscdmstag.h>

static PetscErrorCode FlucaViewerCGNSWriteStructuredSolution_Private(DM dm, Vec v, int file_num, int base, int zone, int sol, const char *name)
{
  PetscInt             dim, xs[3], xm[3], d, cnt, i, j, k, ielem;
  const PetscScalar ***arr2d, ****arr3d;
  cgsize_t             rmin[3], rmax[3], rsize;
  int                  field;
  double              *arrraw;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetCorners(dm, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2], NULL, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, 0, &ielem));

  rsize = 1;
  for (d = 0; d < dim; ++d) {
    rmin[d] = xs[d] + 1;
    rmax[d] = xs[d] + xm[d];
    rsize *= xm[d];
  }

  PetscCall(PetscMalloc1(rsize, &arrraw));
  switch (dim) {
  case 2:
    PetscCall(DMStagVecGetArrayRead(dm, v, &arr2d));
    cnt = 0;
    for (j = rmin[1] - 1; j <= rmax[1] - 1; ++j)
      for (i = rmin[0] - 1; i <= rmax[0] - 1; ++i) {
        arrraw[cnt] = arr2d[j][i][ielem];
        ++cnt;
      }
    PetscCall(DMStagVecRestoreArrayRead(dm, v, &arr2d));
    break;
  case 3:
    PetscCall(DMStagVecGetArrayRead(dm, v, &arr3d));
    cnt = 0;
    for (k = rmin[2] - 1; k <= rmax[2] - 1; ++k)
      for (j = rmin[1] - 1; j <= rmax[1] - 1; ++j)
        for (i = rmin[0] - 1; i <= rmax[0] - 1; ++i) {
          arrraw[cnt] = arr3d[k][j][i][ielem];
          ++cnt;
        }
    PetscCall(DMStagVecRestoreArrayRead(dm, v, &arr3d));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported mesh dimension");
  }
  CGNSCall(cgp_field_write(file_num, base, zone, sol, CGNS_ENUMV(RealDouble), name, &field));
  CGNSCall(cgp_field_write_data(file_num, base, zone, sol, field, rmin, rmax, arrraw));
  PetscCall(PetscFree(arrraw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolView_FSMCGNS(Sol sol, PetscViewer viewer)
{
  Sol_FSM               *fsm = (Sol_FSM *)sol->data;
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;
  DM                     dm;
  PetscInt               dim, d;
  PetscReal              time;
  PetscInt               step;
  PetscReal             *time_slot;
  size_t                *step_slot;
  char                   solution_name[PETSC_MAX_PATH_LEN];
  int                    solution;
  PetscBool              iscart;
  const char *const      velnames[3]      = {"VelocityX", "VelocityY", "VelocityZ"};
  const char *const      intervelnames[3] = {"IntermediateVelocityX", "IntermediateVelocityY", "IntermediateVelocityZ"};
  const char *const      convecnames[3]   = {"ConvectionX", "ConvectionY", "ConvectionZ"};

  PetscFunctionBegin;
  if (!cgv->file_num || !cgv->base) PetscCall(MeshView(sol->mesh, viewer));

  PetscCall(MeshGetDM(sol->mesh, &dm));
  PetscCall(MeshGetDim(sol->mesh, &dim));

  if (!cgv->output_times) PetscCall(PetscSegBufferCreate(sizeof(PetscReal), 20, &cgv->output_times));
  if (!cgv->output_steps) PetscCall(PetscSegBufferCreate(sizeof(size_t), 20, &cgv->output_steps));
  PetscCall(DMGetOutputSequenceNumber(dm, &step, &time));
  if (time < 0.0) {
    step = 0;
    time = 0.0;
  }
  PetscCall(PetscSegBufferGet(cgv->output_times, 1, &time_slot));
  *time_slot = time;
  PetscCall(PetscSegBufferGet(cgv->output_steps, 1, &step_slot));
  *step_slot = step;
  PetscCall(PetscSNPrintf(solution_name, sizeof(solution_name), "FlowSolution%" PetscInt_FMT, step));
  CGNSCall(cg_sol_write(cgv->file_num, cgv->base, cgv->zone, solution_name, CGNS_ENUMV(CellCenter), &solution));

  PetscCall(PetscObjectTypeCompare((PetscObject)sol->mesh, MESHCART, &iscart));
  if (iscart) {
    for (d = 0; d < dim; ++d) {
      PetscCall(FlucaViewerCGNSWriteStructuredSolution_Private(dm, sol->v[d], cgv->file_num, cgv->base, cgv->zone, solution, velnames[d]));
      PetscCall(FlucaViewerCGNSWriteStructuredSolution_Private(dm, fsm->v_star[d], cgv->file_num, cgv->base, cgv->zone, solution, intervelnames[d]));
      PetscCall(FlucaViewerCGNSWriteStructuredSolution_Private(dm, fsm->N[d], cgv->file_num, cgv->base, cgv->zone, solution, convecnames[d]));
    }
    PetscCall(FlucaViewerCGNSWriteStructuredSolution_Private(dm, sol->p, cgv->file_num, cgv->base, cgv->zone, solution, "Pressure"));
    PetscCall(FlucaViewerCGNSWriteStructuredSolution_Private(dm, fsm->p_half, cgv->file_num, cgv->base, cgv->zone, solution, "PressureHalfStep"));
    PetscCall(FlucaViewerCGNSWriteStructuredSolution_Private(dm, fsm->p_prime, cgv->file_num, cgv->base, cgv->zone, solution, "PressureCorrection"));
  }

  PetscCall(PetscViewerFlucaCGNSCheckBatch_Internal(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
