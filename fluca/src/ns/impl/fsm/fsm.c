#include <fluca/private/ns_fsm.h>

extern PetscErrorCode NSFSMInterpolateVelocity2d_MeshCartesian(NS ns);
extern PetscErrorCode NSFSMCalculateConvection2d_MeshCartesian(NS ns);
extern PetscErrorCode NSFSMCalculateIntermediateVelocity2d_MeshCartesian(NS ns);
extern PetscErrorCode NSFSMCalculatePressureCorrection2d_MeshCartesian(NS ns);
extern PetscErrorCode NSFSMUpdate2d_MeshCartesian(NS ns);

PetscErrorCode NSSetup_FSM(NS ns) {
    NS_FSM *fsm = (NS_FSM *)ns->data;
    MPI_Comm comm;
    DM dm;
    PC pc;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    PetscCall(PetscObjectGetComm((PetscObject)ns, &comm));

    /* Create solution */
    PetscCall(SolCreate(comm, &ns->sol));
    PetscCall(SolSetType(ns->sol, SOLFSM));
    PetscCall(SolSetMesh(ns->sol, ns->mesh));

    /* Create KSP */
    PetscCall(MeshGetDM(ns->mesh, &dm));
    PetscCall(KSPCreate(comm, &fsm->ksp));
    PetscCall(KSPSetDM(fsm->ksp, dm));
    PetscCall(KSPGetPC(fsm->ksp, &pc));
    PetscCall(PCSetType(pc, PCMG));
    PetscCall(KSPSetFromOptions(fsm->ksp));

    ns->state = NS_STATE_SETUP;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSSolve_FSM(NS ns, PetscInt num_iters) {
    PetscReal t_init;
    PetscInt dim, i;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(ns, NS_CLASSID, 1);

    if (ns->state < NS_STATE_SETUP)
        PetscCall(NSSetUp(ns));

    t_init = ns->t;
    PetscCall(MeshGetDim(ns->mesh, &dim));
    switch (dim) {
        case 2:
            PetscCall(NSFSMInterpolateVelocity2d_MeshCartesian(ns));
            PetscCall(NSFSMCalculateConvection2d_MeshCartesian(ns));
            for (i = 0; i < num_iters; i++) {
                PetscCall(NSFSMCalculateIntermediateVelocity2d_MeshCartesian(ns));
                PetscCall(NSFSMCalculatePressureCorrection2d_MeshCartesian(ns));
                PetscCall(NSFSMUpdate2d_MeshCartesian(ns));
                ns->step++;
                ns->t = t_init + ns->step * ns->dt;
            }
            break;
        // TODO: consider 3d case
        default:
            SETERRQ(PetscObjectComm((PetscObject)ns), PETSC_ERR_SUP, "Unsupported mesh dimension %" PetscInt_FMT, dim);
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSDestroy_FSM(NS ns) {
    NS_FSM *fsm = (NS_FSM *)ns->data;

    PetscFunctionBegin;

    PetscCall(KSPDestroy(&fsm->ksp));

    PetscCall(PetscFree(ns->data));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSView_FSM(NS ns, PetscViewer v) {
    (void)ns;
    (void)v;

    PetscFunctionBegin;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode NSCreate_FSM(NS ns) {
    NS_FSM *fsm;

    PetscFunctionBegin;

    PetscCall(PetscNew(&fsm));
    ns->data = (void *)fsm;

    fsm->ksp = NULL;

    ns->ops->setup = NSSetup_FSM;
    ns->ops->solve = NSSolve_FSM;
    ns->ops->destroy = NSDestroy_FSM;
    ns->ops->view = NSView_FSM;

    PetscFunctionReturn(PETSC_SUCCESS);
}
