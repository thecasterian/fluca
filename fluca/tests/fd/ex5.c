#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

static const char help[] = "Test FlucaFDApply with boundary value support\n"
                           "Options:\n"
                           "  -input_loc <left|element>   : Input stencil location (default: element)\n"
                           "  -output_loc <left|element>  : Output stencil location (default: element)\n";

static PetscErrorCode CreateDMForLocation(MPI_Comm comm, PetscInt M, DMStagStencilLocation loc, DM *dm)
{
  PetscInt dof0, dof1;

  PetscFunctionBegin;
  switch (loc) {
  case DMSTAG_LEFT:
    dof0 = 1;
    dof1 = 0;
    break;
  case DMSTAG_ELEMENT:
    dof0 = 0;
    dof1 = 1;
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Unsupported stencil location");
  }
  PetscCall(DMStagCreate1d(comm, DM_BOUNDARY_NONE, M, dof0, dof1, DMSTAG_STENCIL_BOX, 1, NULL, dm));
  PetscCall(DMSetUp(*dm));
  PetscCall(DMStagSetUniformCoordinatesProduct(*dm, 0., 1., 0., 0., 0., 0.));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMatrixFromDMs(DM input_dm, DM output_dm, Mat *mat)
{
  PetscInt               input_entries, output_entries;
  ISLocalToGlobalMapping input_ltog, output_ltog;
  MatType                mat_type;

  PetscFunctionBegin;
  PetscCall(DMStagGetEntries(input_dm, &input_entries));
  PetscCall(DMStagGetEntries(output_dm, &output_entries));
  PetscCall(DMGetLocalToGlobalMapping(input_dm, &input_ltog));
  PetscCall(DMGetLocalToGlobalMapping(output_dm, &output_ltog));
  PetscCall(DMGetMatType(output_dm, &mat_type));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)input_dm), mat));
  PetscCall(MatSetSizes(*mat, output_entries, input_entries, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(*mat, mat_type));
  PetscCall(MatSetLocalToGlobalMapping(*mat, output_ltog, input_ltog));
  PetscCall(MatSetUp(*mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FillInputVector(DM dm, DMStagStencilLocation loc, Vec v)
{
  Vec                 v_local;
  PetscInt            x, m, nExtrax, i, slot, slot_coord;
  PetscScalar       **arr;
  const PetscScalar **arr_coord;

  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dm, &v_local));
  PetscCall(VecZeroEntries(v_local));

  PetscCall(DMStagGetCorners(dm, &x, NULL, NULL, &m, NULL, NULL, &nExtrax, NULL, NULL));
  PetscCall(DMStagGetLocationSlot(dm, loc, 0, &slot));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm, loc, &slot_coord));

  PetscCall(DMStagVecGetArray(dm, v_local, &arr));
  /* Fill input vector with f(x) = x */
  if (loc == DMSTAG_LEFT)
    for (i = x; i < x + m + nExtrax; ++i) arr[i][slot] = arr_coord[i][slot_coord];
  else
    for (i = x; i < x + m; ++i) arr[i][slot] = arr_coord[i][slot_coord];
  PetscCall(DMStagVecRestoreArray(dm, v_local, &arr));

  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm, &arr_coord, NULL, NULL));
  PetscCall(DMLocalToGlobal(dm, v_local, INSERT_VALUES, v));
  PetscCall(DMRestoreLocalVector(dm, &v_local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ValidateOutput(Vec output)
{
  Vec       expected;
  PetscReal max_error;

  PetscFunctionBegin;
  /* Create expected vector (all 1's) and compute difference */
  PetscCall(VecDuplicate(output, &expected));
  PetscCall(VecSet(expected, 1.));
  PetscCall(VecAXPY(expected, -1., output));
  PetscCall(VecNorm(expected, NORM_INFINITY, &max_error));
  PetscCall(VecDestroy(&expected));
  PetscCheck(max_error < 1e-10, PetscObjectComm((PetscObject)output), PETSC_ERR_PLIB, "Validation failed: max error %g exceeds tolerance", (double)max_error);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM                    input_dm, output_dm, cdm;
  FlucaFD               fd;
  Mat                   op;
  Vec                   vbc, input, output;
  DMStagStencilLocation input_loc  = DMSTAG_ELEMENT;
  DMStagStencilLocation output_loc = DMSTAG_ELEMENT;
  const PetscInt        M          = 8;

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-input_loc", DMStagStencilLocations, (PetscEnum *)&input_loc, NULL));
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-output_loc", DMStagStencilLocations, (PetscEnum *)&output_loc, NULL));

  PetscCall(CreateDMForLocation(PETSC_COMM_WORLD, M, input_loc, &input_dm));
  if (input_loc == output_loc) {
    output_dm = input_dm;
    PetscCall(PetscObjectReference((PetscObject)output_dm));
  } else {
    PetscCall(CreateDMForLocation(PETSC_COMM_WORLD, M, output_loc, &output_dm));
  }

  PetscCall(DMGetCoordinateDM(input_dm, &cdm));
  PetscCall(FlucaFDDerivativeCreate(cdm, FLUCAFD_X, 1, 1, input_loc, 0, output_loc, 0, &fd));
  /* Call SetFromOptions first so user can set BC types via command line */
  PetscCall(FlucaFDSetFromOptions(fd));

  /*
    Now set boundary VALUES based on the BC types that were set
    For f(x) = x on [0, 1]:
    - If Dirichlet: boundary value = function value (left=0, right=1)
    - If Neumann: boundary value = derivative value = 1
  */
  {
    FlucaFDBoundaryCondition bcs[2];

    PetscCall(FlucaFDGetBoundaryConditions(fd, bcs));
    switch (bcs[0].type) {
    case FLUCAFD_BC_DIRICHLET:
      bcs[0].value = 0.0; /* f(0) = 0 */
      break;
    case FLUCAFD_BC_NEUMANN:
      bcs[0].value = 1.0; /* f'(0) = 1 */
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported BC type");
    }
    switch (bcs[1].type) {
    case FLUCAFD_BC_DIRICHLET:
      bcs[1].value = 1.0; /* f(1) = 1 */
      break;
    case FLUCAFD_BC_NEUMANN:
      bcs[1].value = 1.0; /* f'(1) = 1 */
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported BC type");
    }
    PetscCall(FlucaFDSetBoundaryConditions(fd, bcs));
  }
  PetscCall(FlucaFDSetUp(fd));

  PetscCall(CreateMatrixFromDMs(input_dm, output_dm, &op));
  PetscCall(MatZeroEntries(op));

  PetscCall(DMCreateGlobalVector(output_dm, &vbc));
  PetscCall(VecZeroEntries(vbc));

  PetscCall(FlucaFDApply(fd, input_dm, output_dm, op, vbc));
  PetscCall(MatAssemblyBegin(op, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(vbc));
  PetscCall(MatAssemblyEnd(op, MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyEnd(vbc));
  PetscCall(MatViewFromOptions(op, NULL, "-op_view"));
  PetscCall(VecViewFromOptions(vbc, NULL, "-vbc_view"));

  PetscCall(DMCreateGlobalVector(input_dm, &input));
  PetscCall(FillInputVector(input_dm, input_loc, input));
  PetscCall(DMCreateGlobalVector(output_dm, &output));

  PetscCall(MatMultAdd(op, input, vbc, output));
  PetscCall(VecViewFromOptions(output, NULL, "-output_view"));

  PetscCall(ValidateOutput(output));

  PetscCall(VecDestroy(&input));
  PetscCall(VecDestroy(&output));
  PetscCall(VecDestroy(&vbc));
  PetscCall(MatDestroy(&op));
  PetscCall(FlucaFDDestroy(&fd));
  PetscCall(DMDestroy(&output_dm));
  PetscCall(DMDestroy(&input_dm));

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: elem_to_elem_dirichlet
    nsize: 1
    args: -input_loc element -output_loc element -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_left_bc_type dirichlet -flucafd_right_bc_type dirichlet
    output_file: output/empty.out

  test:
    suffix: elem_to_elem_neumann
    nsize: 1
    args: -input_loc element -output_loc element -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_left_bc_type neumann -flucafd_right_bc_type neumann
    output_file: output/empty.out

  test:
    suffix: elem_to_left_dirichlet
    nsize: 1
    args: -input_loc element -output_loc left -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_left_bc_type dirichlet -flucafd_right_bc_type dirichlet
    output_file: output/empty.out

  test:
    suffix: left_to_elem_dirichlet
    nsize: 1
    args: -input_loc left -output_loc element -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_left_bc_type dirichlet -flucafd_right_bc_type dirichlet
    output_file: output/empty.out

  test:
    suffix: left_to_left_neumann
    nsize: 1
    args: -input_loc left -output_loc left -flucafd_deriv_order 1 -flucafd_accu_order 2 -flucafd_left_bc_type neumann -flucafd_right_bc_type neumann
    output_file: output/empty.out

TEST*/
