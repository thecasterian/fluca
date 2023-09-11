#include <impl/meshimpl.h>
#include <sol/impl/fsm/fsm.h>

extern PetscErrorCode SolView_FSMCGNSCartesian(Sol, PetscViewer);

PetscErrorCode SolSetMesh_FSM(Sol sol, Mesh mesh) {
    Sol_FSM *fsm = (Sol_FSM *)sol->data;
    PetscInt dim;
    DM dm, fdm;

    PetscFunctionBegin;

    PetscCall(VecDestroy(&fsm->p_half));
    PetscCall(VecDestroy(&fsm->UVW));
    PetscCall(VecDestroy(&fsm->u_star));
    PetscCall(VecDestroy(&fsm->v_star));
    PetscCall(VecDestroy(&fsm->w_star));
    PetscCall(VecDestroy(&fsm->UVW_star));
    PetscCall(VecDestroy(&fsm->p_prime));
    PetscCall(VecDestroy(&fsm->Nu));
    PetscCall(VecDestroy(&fsm->Nv));
    PetscCall(VecDestroy(&fsm->Nw));
    PetscCall(VecDestroy(&fsm->p_half_prev));
    PetscCall(VecDestroy(&fsm->Nu_prev));
    PetscCall(VecDestroy(&fsm->Nv_prev));
    PetscCall(VecDestroy(&fsm->Nw_prev));

    PetscCall(VecDestroy(&fsm->u_tilde));
    PetscCall(VecDestroy(&fsm->v_tilde));
    PetscCall(VecDestroy(&fsm->w_tilde));

    PetscCall(MeshGetDim(mesh, &dim));
    PetscCall(MeshGetDM(mesh, &dm));
    PetscCall(MeshGetFaceDM(mesh, &fdm));

    PetscCall(DMCreateLocalVector(dm, &fsm->p_half));
    PetscCall(DMCreateGlobalVector(fdm, &fsm->UVW));
    PetscCall(VecDuplicate(fsm->p_half, &fsm->u_star));
    PetscCall(VecDuplicate(fsm->p_half, &fsm->v_star));
    if (dim > 2)
        PetscCall(VecDuplicate(fsm->p_half, &fsm->w_star));
    PetscCall(VecDuplicate(fsm->UVW, &fsm->UVW_star));
    PetscCall(VecDuplicate(fsm->p_half, &fsm->p_prime));
    PetscCall(VecDuplicate(fsm->p_half, &fsm->Nu));
    PetscCall(VecDuplicate(fsm->p_half, &fsm->Nv));
    if (dim > 2)
        PetscCall(VecDuplicate(fsm->p_half, &fsm->Nw));
    PetscCall(VecDuplicate(fsm->p_half, &fsm->p_half_prev));
    PetscCall(VecDuplicate(fsm->p_half, &fsm->Nu_prev));
    PetscCall(VecDuplicate(fsm->p_half, &fsm->Nv_prev));
    if (dim > 2)
        PetscCall(VecDuplicate(fsm->p_half, &fsm->Nw_prev));

    PetscCall(VecDuplicate(fsm->p_half, &fsm->u_tilde));
    PetscCall(VecDuplicate(fsm->p_half, &fsm->v_tilde));
    if (dim > 2)
        PetscCall(VecDuplicate(fsm->p_half, &fsm->w_tilde));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolDestroy_FSM(Sol sol) {
    Sol_FSM *fsm = (Sol_FSM *)sol->data;

    PetscFunctionBegin;

    PetscCall(VecDestroy(&fsm->p_half));
    PetscCall(VecDestroy(&fsm->UVW));
    PetscCall(VecDestroy(&fsm->u_star));
    PetscCall(VecDestroy(&fsm->v_star));
    PetscCall(VecDestroy(&fsm->w_star));
    PetscCall(VecDestroy(&fsm->UVW_star));
    PetscCall(VecDestroy(&fsm->p_prime));
    PetscCall(VecDestroy(&fsm->Nu));
    PetscCall(VecDestroy(&fsm->Nv));
    PetscCall(VecDestroy(&fsm->Nw));
    PetscCall(VecDestroy(&fsm->p_half_prev));
    PetscCall(VecDestroy(&fsm->Nu_prev));
    PetscCall(VecDestroy(&fsm->Nv_prev));
    PetscCall(VecDestroy(&fsm->Nw_prev));

    PetscCall(VecDestroy(&fsm->u_tilde));
    PetscCall(VecDestroy(&fsm->v_tilde));
    PetscCall(VecDestroy(&fsm->w_tilde));

    PetscCall(PetscFree(sol->data));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolView_FSM(Sol sol, PetscViewer v) {
    PetscBool iscgns, iscart;

    PetscFunctionBegin;

    PetscCall(PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERCGNS, &iscgns));
    PetscCall(PetscObjectTypeCompare((PetscObject)sol->mesh, MESHCARTESIAN, &iscart));
    if (iscgns) {
        if (iscart)
            PetscCall(SolView_FSMCGNSCartesian(sol, v));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SolCreate_FSM(Sol sol) {
    Sol_FSM *fsm;

    PetscFunctionBegin;

    PetscCall(PetscNew(&fsm));
    sol->data = (void *)fsm;

    fsm->p_half = NULL;
    fsm->UVW = NULL;
    fsm->u_star = NULL;
    fsm->v_star = NULL;
    fsm->w_star = NULL;
    fsm->UVW_star = NULL;
    fsm->p_prime = NULL;
    fsm->Nu = NULL;
    fsm->Nv = NULL;
    fsm->Nw = NULL;
    fsm->p_half_prev = NULL;
    fsm->Nu_prev = NULL;
    fsm->Nv_prev = NULL;
    fsm->Nw_prev = NULL;

    fsm->u_tilde = NULL;
    fsm->v_tilde = NULL;
    fsm->w_tilde = NULL;

    sol->ops->setmesh = SolSetMesh_FSM;
    sol->ops->destroy = SolDestroy_FSM;
    sol->ops->view = SolView_FSM;

    PetscFunctionReturn(PETSC_SUCCESS);
}
