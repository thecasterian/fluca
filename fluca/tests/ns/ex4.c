#include <flucans.h>
#include <flucaphys.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test the NS diagnostics (divergence norm, kinetic energy, maximum CFL) on analytic fields\n"
                           "Options:\n"
                           "  -dim <int>   : Dimension, 2 or 3\n"
                           "  -nonuniform  : Stretch the x-direction as x = (i/N)^2\n";

/* A PHYSLAMINAR Phys with density 2 on mesh's DM; every non-periodic side gets a zero-velocity BC */
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
  PetscCall(PhysSetDensity(*phys, 2.));
  for (d = 0; d < dim; ++d) {
    if (bt[d] == DM_BOUNDARY_PERIODIC) continue;
    PetscCall(PhysSetBoundaryCondition(*phys, 2 * d, bc));
    PetscCall(PhysSetBoundaryCondition(*phys, 2 * d + 1, bc));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* x = (i/N)^2 at the x-faces; cell widths 1/16, 3/16, 5/16, 7/16 for N = 4 */
static PetscErrorCode StretchX_Private(DM dm)
{
  PetscScalar **arr[3] = {NULL, NULL, NULL};
  PetscInt      N, gx, gm, iprev, ielem, i;

  PetscFunctionBeginUser;
  PetscCall(DMStagGetGlobalSizes(dm, &N, NULL, NULL));
  PetscCall(DMStagGetGhostCorners(dm, &gx, NULL, NULL, &gm, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateArrays(dm, &arr[0], &arr[1], &arr[2]));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &iprev));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_ELEMENT, &ielem));
  for (i = gx; i < gx + gm; ++i) {
    arr[0][i][iprev] = ((PetscReal)i / N) * ((PetscReal)i / N);
    arr[0][i][ielem] = 0.5 * (((PetscReal)i / N) * ((PetscReal)i / N) + ((PetscReal)(i + 1) / N) * ((PetscReal)(i + 1) / N));
  }
  PetscCall(DMStagRestoreProductCoordinateArrays(dm, &arr[0], &arr[1], &arr[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Cell velocity u = (1, 2[, 3]); face velocity U = x on the x-faces and zero on the others */
static PetscErrorCode SetFields_Private(NS ns)
{
  const PetscScalar **arrc[3] = {NULL, NULL, NULL};
  const PetscScalar   vel[3]  = {1., 2., 3.};
  Phys                phys;
  DM                  dm;
  Vec                 sol;
  DMStagStencil       st;
  PetscScalar         v;
  PetscInt            dim, c_vel, c_U, xs[3], xm[3], nextra[3], slot_prev, i, j, k, d;

  PetscFunctionBeginUser;
  PetscCall(NSGetPhys(ns, &phys));
  PetscCall(PhysGetSolutionDM(phys, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PhysGetField(phys, PHYS_FIELD_VELOCITY, NULL, &c_vel, NULL));
  PetscCall(PhysGetField(phys, PHYS_FIELD_FACE_VELOCITY, NULL, &c_U, NULL));
  PetscCall(NSGetSolution(ns, &sol));
  PetscCall(VecZeroEntries(sol));
  PetscCall(DMStagGetCorners(dm, &xs[0], &xs[1], &xs[2], &xm[0], &xm[1], &xm[2], &nextra[0], &nextra[1], &nextra[2]));
  if (dim < 3) {
    xs[2]     = 0;
    xm[2]     = 1;
    nextra[2] = 0;
  }
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, DMSTAG_LEFT, &slot_prev));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  for (k = xs[2]; k < xs[2] + xm[2]; ++k) {
    for (j = xs[1]; j < xs[1] + xm[1]; ++j) {
      for (i = xs[0]; i < xs[0] + xm[0] + nextra[0]; ++i) {
        st.i = i;
        st.j = j;
        st.k = k;
        if (i < xs[0] + xm[0]) {
          st.loc = DMSTAG_ELEMENT;
          for (d = 0; d < dim; ++d) {
            st.c = c_vel + d;
            v    = vel[d];
            PetscCall(DMStagVecSetValuesStencil(dm, sol, 1, &st, &v, INSERT_VALUES));
          }
        }
        st.loc = DMSTAG_LEFT;
        st.c   = c_U;
        v      = arrc[0][i][slot_prev];
        PetscCall(DMStagVecSetValuesStencil(dm, sol, 1, &st, &v, INSERT_VALUES));
      }
    }
  }
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arrc[0], &arrc[1], &arrc[2]));
  PetscCall(VecAssemblyBegin(sol));
  PetscCall(VecAssemblyEnd(sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM             dm;
  Mesh           mesh;
  Phys           phys;
  NS             ns;
  PetscInt       dim        = 2, n;
  PetscBool      nonuniform = PETSC_FALSE;
  PetscReal      nrm, ke, cfl;
  const NormType types[3] = {NORM_1, NORM_2, NORM_INFINITY};

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nonuniform", &nonuniform, NULL));
  if (dim == 2) PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, &dm));
  else PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, 4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 0, 0, 0, 1, DMSTAG_STENCIL_STAR, 2, NULL, NULL, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm, 0., 1., 0., 1., 0., 1.));
  if (nonuniform) PetscCall(StretchX_Private(dm));

  PetscCall(MeshCartesianCreate(dm, &mesh));
  PetscCall(MeshSetFromOptions(mesh));
  PetscCall(MeshSetUp(mesh));

  PetscCall(CreatePhys_Private(mesh, &phys));
  PetscCall(NSCreate(PETSC_COMM_WORLD, &ns));
  PetscCall(NSSetPhys(ns, phys));
  PetscCall(NSSetTimeStepSize(ns, 0.1));
  PetscCall(NSSetFromOptions(ns));
  PetscCall(NSSetUp(ns));
  PetscCall(SetFields_Private(ns));

  for (n = 0; n < 3; ++n) {
    PetscCall(NSComputeDivergenceNorm(ns, types[n], &nrm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Divergence norm %s: %g\n", NormTypes[types[n]], (double)nrm));
  }
  PetscCall(NSComputeKineticEnergy(ns, &ke));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Kinetic energy: %g\n", (double)ke));
  PetscCall(NSComputeMaxCFL(ns, &cfl));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Max CFL: %g\n", (double)cfl));

  PetscCall(NSDestroy(&ns));
  PetscCall(PhysDestroy(&phys));
  PetscCall(MeshDestroy(&mesh));
  PetscCall(DMDestroy(&dm));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d
    nsize: 1

  test:
    suffix: 2d_nonuniform
    nsize: 1
    args: -nonuniform

  test:
    suffix: 2d_periodic
    nsize: 1
    args: -stag_boundary_type_x periodic

  test:
    suffix: 3d
    nsize: 1
    args: -dim 3

TEST*/
