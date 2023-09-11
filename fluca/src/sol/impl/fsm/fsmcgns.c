#include <flucamap.h>
#include <impl/viewercgnsutils.h>
#include <petsc/private/viewercgnsimpl.h>
#include <sol/impl/fsm/fsm.h>

PetscErrorCode SolView_FSMCGNSCartesian(Sol sol, PetscViewer v) {
    Sol_FSM *fsm = (Sol_FSM *)sol->data;
    PetscViewer_CGNS *cgns = (PetscViewer_CGNS *)v->data;
    PetscInt dim;
    FlucaMap map;
    PetscContainer viewerinfo_container;
    ViewerCGNSInfo *viewerinfo;
    DM dm;

    PetscFunctionBegin;

    PetscCall(MeshGetDim(sol->mesh, &dim));

    PetscCall(PetscObjectQuery((PetscObject)v, "_Fluca_CGNSViewerMeshCartesianMap", (PetscObject *)&map));
    PetscCall(FlucaMapGetValue(map, (PetscObject)sol->mesh, (PetscObject *)&viewerinfo_container));
    PetscCall(PetscContainerGetPointer(viewerinfo_container, (void **)&viewerinfo));

    PetscCall(MeshGetDM(sol->mesh, &dm));
    PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->p_half, cgns->file_num, cgns->base, viewerinfo->zone,
                                                        viewerinfo->sol, "PressureHalfStep"));
    PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->u_star, cgns->file_num, cgns->base, viewerinfo->zone,
                                                        viewerinfo->sol, "IntermediateVelocityX"));
    PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->v_star, cgns->file_num, cgns->base, viewerinfo->zone,
                                                        viewerinfo->sol, "IntermediateVelocityY"));
    if (dim > 2)
        PetscCall(ViewerCGNSWriteStructuredSolution_Private(
            dm, fsm->w_star, cgns->file_num, cgns->base, viewerinfo->zone, viewerinfo->sol, "IntermediateVelocityZ"));
    PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->p_prime, cgns->file_num, cgns->base, viewerinfo->zone,
                                                        viewerinfo->sol, "PressureCorrection"));
    PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->Nu, cgns->file_num, cgns->base, viewerinfo->zone,
                                                        viewerinfo->sol, "ConvectionX"));
    PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->Nv, cgns->file_num, cgns->base, viewerinfo->zone,
                                                        viewerinfo->sol, "ConvectionY"));
    if (dim > 2)
        PetscCall(ViewerCGNSWriteStructuredSolution_Private(dm, fsm->Nw, cgns->file_num, cgns->base, viewerinfo->zone,
                                                            viewerinfo->sol, "ConvectionZ"));

    PetscFunctionReturn(PETSC_SUCCESS);
}
