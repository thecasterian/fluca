#include <fluca/private/solimpl.h>
#include <fluca/private/flucaviewer_cgns.h>
#include <pcgnslib.h>

PetscErrorCode SolLoadCGNS(Sol sol, PetscInt file_num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  PetscCheck(sol->mesh, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Mesh is not set");
  PetscTryTypeMethod(sol, loadcgns, file_num);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolLoadCGNSFromFile(Sol sol, const char filename[])
{
  int file_num = -1;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sol, SOL_CLASSID, 1);
  PetscAssertPointer(filename, 2);

  CGNSCall(cgp_mpi_comm(PetscObjectComm((PetscObject)sol)));
  CGNSCall(cgp_open(filename, CG_MODE_READ, &file_num));
  PetscCheck(file_num > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cgp_open(\"%s\", ...) did not return a valid file number", filename);
  PetscCall(SolLoadCGNS(sol, file_num));
  CGNSCall(cgp_close(file_num));
  PetscFunctionReturn(PETSC_SUCCESS);
}
