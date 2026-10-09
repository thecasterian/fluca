#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test the pressure null space NS derives from the boundary condition types\n";

/* A PHYSLAMINAR Phys on mesh's DM; every non-periodic side gets a zero-velocity BC */
static PetscErrorCode CreatePhys_Private(Mesh mesh, Phys *phys)
{
  DM             dm;
  DMBoundaryType bt[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  PhysBC         bc    = {PHYS_BC_VELOCITY, NULL, NULL, NULL, NULL};
  PetscInt       dim, d;

  PetscFunctionBeginUser;
  PetscCall(MeshGetDM(mesh, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMStagGetBoundaryTypes(dm, &bt[0], &bt[1], &bt[2]));
  PetscCall(PhysCreate(PetscObjectComm((PetscObject)dm), phys));
  PetscCall(PhysSetType(*phys, PHYSLAMINAR));
  PetscCall(PhysSetMesh(*phys, mesh));
  for (d = 0; d < dim; ++d) {
    if (bt[d] == DM_BOUNDARY_PERIODIC) continue;
    PetscCall(PhysSetBoundaryCondition(*phys, 2 * d, bc));
    PetscCall(PhysSetBoundaryCondition(*phys, 2 * d + 1, bc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM           dm;
  Mesh         mesh;
  Phys         phys;
  NS           ns;
  SNES         snes;
  Mat          J;
  MatNullSpace nsp;
  PetscBool    has_const;
  PetscInt     n, k;
  const Vec   *vecs;
  IS           is;
  Vec          sub;
  PetscReal    nrm, vmin, vmax;
  const char  *names[3] = {PHYS_FIELD_VELOCITY, PHYS_FIELD_FACE_VELOCITY, PHYS_FIELD_PRESSURE};

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 0.));

  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));

  PetscCall(CreatePhys_Private(mesh, &phys));
  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));

  PetscCall(NSGetSNES(ns, &snes));
  PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
  PetscCall(MatGetNullSpace(J, &nsp));
  PetscCall(MatNullSpaceGetVecs(nsp, &has_const, &n, &vecs));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Null space: %" PetscInt_FMT " vector(s), constant %s\n", n, has_const ? "yes" : "no"));
  for (k = 0; k < 3; ++k) {
    PetscCall(NSGetField(ns, names[k], &is));
    PetscCall(VecGetSubVector(vecs[0], is, &sub));
    PetscCall(VecNorm(sub, NORM_2, &nrm));
    PetscCall(VecMin(sub, NULL, &vmin));
    PetscCall(VecMax(sub, NULL, &vmax));
    PetscCall(VecRestoreSubVector(vecs[0], is, &sub));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: norm %g, min %g, max %g\n", names[k], (double)nrm, (double)vmin, (double)vmax));
  }

  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: walled
    nsize: 1

  test:
    suffix: periodic
    nsize: 1
    args: -stag_boundary_type_x periodic -stag_boundary_type_y periodic

TEST*/
