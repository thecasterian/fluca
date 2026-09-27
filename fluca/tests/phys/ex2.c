#include <flucaphys.h>
#include <flucasys.h>
#include <flucaviewer.h>
#include <petscdmstag.h>

static const char help[] = "Test Phys solution vectors: CGNS round trip of every field\n"
                           "Options:\n"
                           "  -file <name> : write the solution to this CGNS file and load it back into a duplicate\n"
                           "  -ascii       : view the solution vector with the ASCII viewer instead\n";

/* Field values at point coordinates: velocity (x + 2y, xy), face velocity (x on x-faces, y on y-faces), pressure x - y */
static PetscErrorCode FillSolution_Private(Phys phys, Vec u)
{
  DM                  dm;
  Vec                 loc;
  PetscScalar      ***arr;
  const PetscScalar **cx, **cy;
  PetscInt            x, y, m, n, i, j, c_vel, c_U, c_p, s_u, s_v, s_p, s_Ul, s_Ud, iprev, ielem;
  PetscBool           lastx, lasty;

  PetscFunctionBegin;
  PetscCall(PhysGetSolutionDM(phys, &dm));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_PRESSURE, NULL, &c_p, NULL));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, c_vel, &s_u));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, c_vel + 1, &s_v));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_ELEMENT, c_p, &s_p));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_LEFT, c_U, &s_Ul));
  PetscCall(DMStagGetLocationSlot(dm, DMSTAG_DOWN, c_U, &s_Ud));
  PetscCall(DMStagGetCorners(dm, &x, &y, NULL, &m, &n, NULL, NULL, NULL, NULL));
  PetscCall(DMStagGetIsLastRank(dm, &lastx, &lasty, NULL));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &cx, &cy, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielem));
  PetscCall(DMGetLocalVector(dm, &loc));
  PetscCall(VecZeroEntries(loc));
  PetscCall(DMStagVecGetArray(dm, loc, &arr));
  for (j = y; j < y + n + (lasty ? 1 : 0); ++j)
    for (i = x; i < x + m + (lastx ? 1 : 0); ++i) {
      if (j < y + n) arr[j][i][s_Ul] = cx[i][iprev];
      if (i < x + m) arr[j][i][s_Ud] = cy[j][iprev];
      if (i < x + m && j < y + n) {
        arr[j][i][s_u] = cx[i][ielem] + 2. * cy[j][ielem];
        arr[j][i][s_v] = cx[i][ielem] * cy[j][ielem];
        arr[j][i][s_p] = cx[i][ielem] - cy[j][ielem];
      }
    }
  PetscCall(DMStagVecRestoreArray(dm, loc, &arr));
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &cx, &cy, NULL));
  PetscCall(DMLocalToGlobal(dm, loc, INSERT_VALUES, u));
  PetscCall(DMRestoreLocalVector(dm, &loc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm, sol_dm;
  Mesh        mesh;
  Phys        phys;
  Vec         u, w, sub;
  IS          is;
  PetscViewer viewer;
  PetscInt    nfields, f, step;
  PetscReal   time, nrm;
  const char *name;
  char        file[PETSC_MAX_PATH_LEN] = "phys_ex2.cgns";
  PetscBool   ascii                    = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-file", file, sizeof(file), NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ascii", &ascii, NULL));

  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 3, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 3., 0., 0.));
  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetUp(mesh));
  PetscCall(PhysCreate(PETSC_COMM_WORLD, &phys));
  PetscCall(PhysSetType(phys, PHYSLAMINAR));
  PetscCall(PhysSetMesh(phys, mesh));
  PetscCall(PhysSetUp(phys));

  PetscCall(PhysCreateSolutionVector(phys, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "Solution"));
  PetscCall(FillSolution_Private(phys, u));

  if (ascii) PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));
  else {
    PetscCall(PhysGetSolutionDM(phys, &sol_dm));
    PetscCall(DMSetOutputSequenceNumber(sol_dm, 3, 0.5));
    PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, file, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(u, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    /* Load into a duplicate: VecDuplicate must carry the view/load ops and the Phys */
    PetscCall(DMSetOutputSequenceNumber(sol_dm, -1, 0.));
    PetscCall(VecDuplicate(u, &w));
    PetscCall(VecZeroEntries(w));
    PetscCall(PetscViewerFlucaCGNSOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &viewer));
    PetscCall(FlucaVecLoad(w, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(DMGetOutputSequenceNumber(sol_dm, &step, &time));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Loaded step %" PetscInt_FMT " time %g\n", step, (double)time));
    PetscCall(PhysGetNumFields(phys, &nfields));
    for (f = 0; f < nfields; ++f) {
      PetscCall(PhysGetFieldName(phys, f, &name));
      PetscCall(PhysGetFieldIS(phys, name, &is));
      PetscCall(VecGetSubVector(w, is, &sub));
      PetscCall(VecNorm(sub, NORM_INFINITY, &nrm));
      PetscCall(VecRestoreSubVector(w, is, &sub));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: max |value| %g\n", name, (double)nrm));
    }
    PetscCall(VecAXPY(w, -1., u));
    PetscCall(VecNorm(w, NORM_INFINITY, &nrm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max |loaded - written|: %g\n", (double)nrm));
    PetscCall(VecDestroy(&w));
  }

  PetscCall(VecDestroy(&u));
  PetscCall(PhysDestroy(&phys));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: cgns_roundtrip
    nsize: 1
    args: -file phys_ex2.cgns

  test:
    suffix: ascii
    nsize: 1
    args: -ascii

TEST*/
