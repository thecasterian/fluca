#include <fluca/private/sol_fsm.h>
#include <fluca/private/viewercgnsutils.h>
#include <flucamap.h>
#include <petsc/private/viewercgnsimpl.h>

PetscErrorCode SolView_FSMCGNSCartesian(Sol sol, PetscViewer v) {
    Sol_FSM *fsm = (Sol_FSM *)sol->data;
    PetscViewer_CGNS *cgns = (PetscViewer_CGNS *)v->data;
    FlucaMap map;
    PetscContainer viewerinfo_container;
    ViewerCGNSInfo *viewerinfo;
    DM dm;
    PetscInt dim, d;
    const char *const intervelnames[3] = {"IntermediateVelocityX", "IntermediateVelocityY", "IntermediateVelocityZ"};
    const char *const convecnames[3] = {"ConvectionX", "ConvectionY", "ConvectionZ"};

    PetscFunctionBegin;

    PetscCall(PetscObjectQuery((PetscObject)v, "_Fluca_CGNSViewerMeshCartesianMap", (PetscObject *)&map));
    PetscCall(FlucaMapGetValue(map, (PetscObject)sol->mesh, (PetscObject *)&viewerinfo_container));
    PetscCall(PetscContainerGetPointer(viewerinfo_container, (void **)&viewerinfo));

    PetscCall(MeshGetDM(sol->mesh, &dm));
    PetscCall(MeshGetDim(sol->mesh, &dim));

    for (d = 0; d < dim; d++) {
        PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->v_star[d], cgns->file_num, cgns->base,
                                                            viewerinfo->zone, viewerinfo->sol, intervelnames[d]));
        PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->N[d], cgns->file_num, cgns->base, viewerinfo->zone,
                                                            viewerinfo->sol, convecnames[d]));
    }
    PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->p_half, cgns->file_num, cgns->base, viewerinfo->zone,
                                                        viewerinfo->sol, "PressureHalfStep"));
    PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->p_prime, cgns->file_num, cgns->base, viewerinfo->zone,
                                                        viewerinfo->sol, "PressureCorrection"));

    PetscFunctionReturn(PETSC_SUCCESS);
}
