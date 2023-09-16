#include <fluca/private/ns_fsm.h>
#include <fluca/private/nsimpl.h>

PetscErrorCode NSSetup_FSM(NS ns) {
    NS_FSM *fsm = (NS_FSM *)ns->data;
    MPI_Comm comm;
    DM dm;
    PetscInt dim, d;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));

    /* Create solution */
    PetscCall(SolCreate(comm, &ns->sol));
    PetscCall(SolSetType(ns->sol, SOLFSM));
    PetscCall(SolSetMesh(ns->sol, ns->mesh));

    /* Create KSP */
    PetscCall(MeshGetDim(ns->mesh, &dim));
    PetscCall(MeshGetDM(ns->mesh, &dm));
    // TODO: set RHS and operators
    for (d = 0; d < dim; d++) {
        PetscCall(KSPCreate(comm, &fsm->kspv[d]));
        PetscCall(KSPSetDM(fsm->kspv[d], dm));
    }
    PetscCall(KSPCreate(comm, &fsm->kspp));
    PetscCall(KSPSetDM(fsm->kspp, dm));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSDestroy_FSM(NS ns) {
    NS_FSM *fsm = (NS_FSM *)ns->data;
    PetscInt d;

    PetscFunctionBegin;

    for (d = 0; d < 3; d++)
        PetscCall(KSPDestroy(&fsm->kspv[d]));
    PetscCall(KSPDestroy(&fsm->kspp));

    PetscCall(PetscFree(ns->data));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSView_FSM(NS ns, PetscViewer v) {
    PetscFunctionBegin;

    // TODO: implement

    (void)ns;
    (void)v;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSCreate_FSM(NS ns) {
    NS_FSM *fsm;
    PetscInt d;

    PetscFunctionBegin;

    PetscCall(PetscNew(&fsm));
    ns->data = (void *)fsm;

    for (d = 0; d < 3; d++)
        fsm->kspv[d] = NULL;
    fsm->kspp = NULL;

    ns->ops->destroy = NSDestroy_FSM;
    ns->ops->view = NSView_FSM;

    PetscFunctionReturn(PETSC_SUCCESS);
}
