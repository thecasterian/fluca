#include <flucaviewer.h>
#include <fluca/private/flucaimpl.h>

static PetscErrorCode FlucaOptionsCreateViewersSingle_Private(MPI_Comm comm, const char value[], PetscViewer *viewer, PetscViewerFormat *format)
{
  char       *value_copy, *filename = NULL, *viewer_format = NULL, *filemode = NULL;
  const char *viewer_type;
  size_t      len;
  const char *viewers[] = {"ascii", "cgns", NULL};
  PetscInt    cnt;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(value, &len));
  if (!len) {
    if (format) *format = PETSC_VIEWER_DEFAULT;
    if (viewer) {
      PetscCall(PetscViewerASCIIGetStdout(comm, viewer));
      PetscCall(PetscObjectReference((PetscObject)*viewer));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrallocpy(value, &value_copy));
  PetscCall(PetscStrchr(value_copy, ':', &filename));
  if (filename) {
    *filename = 0;
    ++filename;
    PetscCall(PetscStrchr(filename, ':', &viewer_format));
  }
  if (viewer_format) {
    *viewer_format = 0;
    ++viewer_format;
    PetscCall(PetscStrchr(viewer_format, ':', &filemode));
  }
  if (filemode) {
    *filemode = 0;
    ++filemode;
  }
  /* If no viewer type is specified, e.g., ":output.txt", use default type ascii. */
  viewer_type = *value_copy ? value_copy : "ascii";
  PetscCall(PetscStrendswithwhich(viewer_type, viewers, &cnt));
  PetscCheck(cnt < (PetscInt)(sizeof(viewers) / sizeof(viewers[0])), comm, PETSC_ERR_ARG_WRONG, "Unsupported viewer type: %s", viewer_type);

  if (viewer) {
    if (!filename || !*filename) {
      switch (cnt) {
      case 0: /* ascii */
        PetscCall(PetscViewerASCIIGetStdout(comm, viewer));
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
      case 1: /* cgns */
        if (!(*viewer = PETSC_VIEWER_FLUCACGNS_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
      default:
        SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unknown viewer type");
      }
    } else {
      PetscFileMode mode = FILE_MODE_WRITE;
      PetscBool     flag = PETSC_FALSE;

      PetscCall(PetscViewerCreate(comm, viewer));
      switch (cnt) {
      case 0: /* ascii */
        PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERASCII));
        break;
      case 1: /* cgns */
        PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERFLUCACGNS));
        break;
      }
      if (filemode && *filemode) {
        PetscCall(PetscEnumFind(PetscFileModes, filemode, (PetscEnum *)&mode, &flag));
        PetscCheck(flag, comm, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown file mode: %s", filemode);
      }
      PetscCall(PetscViewerFileSetMode(*viewer, mode));
      PetscCall(PetscViewerFileSetName(*viewer, filename)); /* Viewer opens file when set name; must set mode first. */
      PetscCall(PetscViewerSetFromOptions(*viewer));
    }
    PetscCall(PetscViewerSetUp(*viewer));
  }

  if (viewer_format && *viewer_format) {
    PetscViewerFormat format;
    PetscBool         flag;

    PetscCall(PetscEnumFind(PetscViewerFormats, viewer_format, (PetscEnum *)&format, &flag));
    PetscCheck(flag, comm, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown viewer format: %s", viewer_format);
  }

  PetscCall(PetscFree(value_copy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaOptionsCreateViewers_Private(MPI_Comm comm, PetscOptions options, const char pre[], const char name[], PetscViewer *viewer, PetscViewerFormat *format, PetscBool *set, const char func_name[])
{
  const char *value;
  PetscBool   flag, has_help;

  PetscFunctionBegin;
  if (set) *set = PETSC_FALSE;
  PetscCall(PetscOptionsGetCreateViewerOff(&flag));
  if (flag) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscOptionsHasHelp(NULL, &has_help));
  if (has_help) {
    if (viewer) {
      PetscCall((*PetscHelpPrintf)(comm, "----------------------------------------\nViewer (-%s%s) options:\n", pre ? pre : "", name + 1));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s ascii[:[filename][:[format][:append]]]: %s (%s)\n", pre ? pre : "", name + 1, "Prints object to stdout or ASCII file", func_name));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s cgns[:filename]: %s (%s)\n", pre ? pre : "", name + 1, "Saves object to a CGNS file", func_name));
    }
  }

  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (flag) {
    if (set) *set = PETSC_TRUE;
    if (!value) {
      if (format) *format = PETSC_VIEWER_DEFAULT;
      if (viewer) {
        PetscCall(PetscViewerASCIIGetStdout(comm, viewer));
        PetscCall(PetscObjectReference((PetscObject)*viewer));
      }
    } else {
      char *comma_separator;

      PetscCall(PetscStrchr(value, ',', &comma_separator));
      PetscCheck(!comma_separator, comm, PETSC_ERR_ARG_WRONG, "Multiple viewers are not supported in %s", func_name);
      PetscCall(FlucaOptionsCreateViewersSingle_Private(comm, value, viewer, format));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaOptionsCreateViewer(MPI_Comm comm, PetscOptions options, const char pre[], const char name[], PetscViewer *viewer, PetscViewerFormat *format, PetscBool *set)
{
  PetscBool set_internal;

  PetscFunctionBegin;
  PetscAssertPointer(name, 4);
  if (viewer) *viewer = NULL;
  if (format) *format = PETSC_VIEWER_DEFAULT;
  if (set) *set = PETSC_FALSE;
  PetscCall(FlucaOptionsCreateViewers_Private(comm, options, pre, name, viewer, format, &set_internal, PETSC_FUNCTION_NAME));
  if (set) *set = set_internal;
  PetscFunctionReturn(PETSC_SUCCESS);
}
