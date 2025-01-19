#include <fluca/private/flucaviewer_cgns.h>
#include <pcgnslib.h>

PetscMPIInt Petsc_Viewer_FlucaCGNS_keyval = MPI_KEYVAL_INVALID;

PetscErrorCode FlucaGetCGNSDataType_Internal(PetscDataType petsc_dtype, CGNS_ENUMT(DataType_t) *cgns_dtype)
{
  PetscFunctionBegin;
  switch (petsc_dtype) {
  case PETSC_DOUBLE:
    *cgns_dtype = CGNS_ENUMV(RealDouble);
    break;
  case PETSC_FLOAT:
    *cgns_dtype = CGNS_ENUMV(RealSingle);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported data type");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileClose_FlucaCGNS_Private(PetscViewer viewer)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  if (!cgv->file_num) PetscFunctionReturn(PETSC_SUCCESS);

  if (cgv->output_times) {
    PetscCount       size, *steps;
    char            *solnames;
    PetscReal       *times;
    cgsize_t         num_times;
    const PetscCount width = 32;

    PetscCall(PetscSegBufferGetSize(cgv->output_times, &size));
    PetscCall(PetscSegBufferExtractInPlace(cgv->output_times, &times));

    num_times = size;
    CGNSCall(cg_biter_write(cgv->file_num, cgv->base, "TimeIterValues", num_times));
    CGNSCall(cg_goto(cgv->file_num, cgv->base, "BaseIterativeData_t", 1, NULL));
    CGNSCall(cg_array_write("TimeValues", CGNS_ENUMV(RealDouble), 1, &num_times, times));
    PetscCall(PetscSegBufferDestroy(&cgv->output_times));
    CGNSCall(cg_ziter_write(cgv->file_num, cgv->base, cgv->zone, "ZoneIterativeData"));
    CGNSCall(cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "ZoneIterativeData_t", 1, NULL));

    PetscCall(PetscMalloc(size * width + 1, &solnames));
    PetscCall(PetscSegBufferExtractInPlace(cgv->output_steps, &steps));
    cgsize_t shape[2] = {(cgsize_t)width, (cgsize_t)size};
    for (PetscCount i = 0; i < size; i++) PetscCall(PetscSNPrintf(&solnames[i * width], width + 1, "FlowSolution%-20zu", (size_t)steps[i]));
    CGNSCall(cg_array_write("FlowSolutionPointers", CGNS_ENUMV(Character), 2, shape, solnames));
    PetscCall(PetscSegBufferDestroy(&cgv->output_steps));
    for (PetscCount i = 0; i < size; i++) PetscCall(PetscSNPrintf(&solnames[i * width], width + 1, "%-32s", "CellInfo"));
    CGNSCall(cg_array_write("FlowSolutionCellInfoPointers", CGNS_ENUMV(Character), 2, shape, solnames));
    PetscCall(PetscFree(solnames));

    CGNSCall(cg_simulation_type_write(cgv->file_num, cgv->base, CGNS_ENUMV(TimeAccurate)));
  }

  PetscCall(PetscFree(cgv->filename));

  CGNSCall(cgp_close(cgv->file_num));
  cgv->file_num = 0;
  cgv->base     = 0;
  cgv->zone     = 0;
  cgv->sol      = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFileOpen_FlucaCGNS_Internal(PetscViewer viewer, PetscInt sequence_number)
{
  PetscViewer_FlucaCGNS *cgv  = (PetscViewer_FlucaCGNS *)viewer->data;
  int                    mode = -1;

  PetscFunctionBegin;
  PetscCheck((cgv->filename == NULL) ^ (sequence_number < 0), PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Expect either a template filename or non-negative sequence number");

  if (!cgv->filename) {
    char filename[PETSC_MAX_PATH_LEN];
    PetscCall(PetscSNPrintf(filename, sizeof filename, cgv->filename_template, (int)sequence_number));
    PetscCall(PetscStrallocpy(filename, &cgv->filename));
  }

  switch (cgv->filemode) {
  case FILE_MODE_READ:
    mode = CG_MODE_READ;
    break;
  case FILE_MODE_WRITE:
    mode = CG_MODE_WRITE;
    break;
  case FILE_MODE_UNDEFINED:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  default:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Unsupported file mode %s", PetscFileModes[cgv->filemode]);
  }

  CGNSCall(cgp_mpi_comm(PetscObjectComm((PetscObject)viewer)));
  CGNSCall(cgp_open(cgv->filename, mode, &cgv->file_num));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFlucaCGNSCheckBatch_Internal(PetscViewer viewer)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;
  PetscCount             num_steps;

  PetscFunctionBegin;
  if (!cgv->filename_template) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscSegBufferGetSize(cgv->output_steps, &num_steps));
  if (num_steps >= (PetscCount)cgv->batch_size) PetscCall(PetscViewerFileClose_FlucaCGNS_Private(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerDestroy_FlucaCGNS(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerFileClose_FlucaCGNS_Private(viewer));
  PetscCall(PetscFree(viewer->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerView_FlucaCGNS(PetscViewer v, PetscViewer viewer)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)v->data;
  PetscBool              isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    if (cgv->filename) PetscCall(PetscViewerASCIIPrintf(viewer, "Filename: %s\n", cgv->filename));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerSetFromOptions_FlucaCGNS(PetscViewer viewer, PetscOptionItems PetscOptionsObject)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "CGNS Viewer Options");
  PetscCall(PetscOptionsBoundedInt("-viewer_cgns_batch_size", "Max number of output sequence times to write per batch", "PetscViewerFlucaCGNSSetBatchSize", cgv->batch_size, &cgv->batch_size, NULL, 1));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetName_FlucaCGNS(PetscViewer viewer, const char filename[])
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;
  char                  *has_pattern;

  PetscFunctionBegin;
  if (cgv->file_num) PetscCall(PetscViewerFileClose_FlucaCGNS_Private(viewer));
  PetscCall(PetscFree(cgv->filename_template));
  PetscCall(PetscFree(cgv->filename));
  PetscCall(PetscStrchr(filename, '%', &has_pattern));
  if (has_pattern) {
    PetscCall(PetscStrallocpy(filename, &cgv->filename_template));
  } else {
    PetscCall(PetscStrallocpy(filename, &cgv->filename));
    PetscCall(PetscViewerFileOpen_FlucaCGNS_Internal(viewer, -1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetName_FlucaCGNS(PetscViewer viewer, const char **filename)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  *filename = cgv->filename;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetMode_FlucaCGNS(PetscViewer viewer, PetscFileMode filemode)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  cgv->filemode = filemode;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetMode_FlucaCGNS(PetscViewer viewer, PetscFileMode *filemode)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  *filemode = cgv->filemode;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerCreate_FlucaCGNS(PetscViewer viewer)
{
  PetscViewer_FlucaCGNS *cgv;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cgv));

  viewer->data                = cgv;
  viewer->ops->destroy        = PetscViewerDestroy_FlucaCGNS;
  viewer->ops->view           = PetscViewerView_FlucaCGNS;
  viewer->ops->setfromoptions = PetscViewerSetFromOptions_FlucaCGNS;
  viewer->ops->setup          = NULL;

  cgv->file_num     = 0;
  cgv->base         = 0;
  cgv->zone         = 0;
  cgv->sol          = 0;
  cgv->output_steps = NULL;
  cgv->output_times = NULL;
  cgv->last_step    = -1;
  cgv->batch_size   = 1;

  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetName_C", PetscViewerFileSetName_FlucaCGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetName_C", PetscViewerFileGetName_FlucaCGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetMode_C", PetscViewerFileSetMode_FlucaCGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", PetscViewerFileGetMode_FlucaCGNS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFlucaCGNSOpen(MPI_Comm comm, const char filename[], PetscFileMode filemode, PetscViewer *viewer)
{
  PetscFunctionBegin;
  PetscAssertPointer(filename, 2);
  PetscAssertPointer(viewer, 4);
  PetscCall(PetscViewerCreate(comm, viewer));
  PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERFLUCACGNS));
  PetscCall(PetscViewerFileSetMode(*viewer, filemode));
  PetscCall(PetscViewerFileSetName(*viewer, filename));
  PetscCall(PetscViewerSetFromOptions(*viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFlucaCGNSSetBatchSize(PetscViewer viewer, PetscInt batch_size)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERFLUCACGNS);
  cgv->batch_size = batch_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerFlucaCGNSGetBatchSize(PetscViewer viewer, PetscInt *batch_size)
{
  PetscViewer_FlucaCGNS *cgv = (PetscViewer_FlucaCGNS *)viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(viewer, PETSC_VIEWER_CLASSID, 1, PETSCVIEWERFLUCACGNS);
  PetscAssertPointer(batch_size, 2);
  *batch_size = cgv->batch_size;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscViewer PETSC_VIEWER_FLUCACGNS_(MPI_Comm comm)
{
  MPI_Comm    ncomm;
  PetscViewer viewer;
  int         mpiflag;
  PetscBool   flag;
  char        fname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscCallNull(PetscCommDuplicate(comm, &ncomm, NULL));
  if (Petsc_Viewer_FlucaCGNS_keyval == MPI_KEYVAL_INVALID) PetscCallMPINull(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, MPI_COMM_NULL_DELETE_FN, &Petsc_Viewer_FlucaCGNS_keyval, NULL));
  PetscCallMPINull(MPI_Comm_get_attr(ncomm, Petsc_Viewer_FlucaCGNS_keyval, (void **)&viewer, (int *)&mpiflag));
  if (!mpiflag) {
    PetscCallNull(PetscOptionsGetenv(ncomm, "PETSC_VIEWER_CGNS_FILENAME", fname, PETSC_MAX_PATH_LEN, &flag));
    if (!flag) PetscCallNull(PetscStrncpy(fname, "output.cgns", sizeof(fname)));
    PetscCallNull(PetscViewerFlucaCGNSOpen(ncomm, fname, FILE_MODE_WRITE, &viewer));
    PetscCallNull(PetscObjectRegisterDestroy((PetscObject)viewer));
    PetscCallMPINull(MPI_Comm_set_attr(ncomm, Petsc_Viewer_FlucaCGNS_keyval, (void *)viewer));
  }
  PetscCallNull(PetscCommDestroy(&ncomm));
  PetscFunctionReturn(viewer);
}
