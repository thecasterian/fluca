#include <fluca/private/solimpl.h>
#include <fluca/private/viewercgnsutils.h>
#include <flucamap.h>
#include <petsc/private/viewercgnsimpl.h>

PetscErrorCode SolView_CGNSCartesian(Sol sol, PetscViewer v) {
    PetscViewer_CGNS *cgns = (PetscViewer_CGNS *)v->data;
    FlucaMap map;
    PetscContainer viewerinfo_container;
    ViewerCGNSInfo *viewerinfo;
    DM dm;
    PetscInt dim, d;
    const char *const velnames[3] = {"VelocityX", "VelocityY", "VelocityZ"};

    PetscFunctionBegin;

    PetscCall(MeshView(sol->mesh, v));

    PetscCall(PetscObjectQuery((PetscObject)v, "_Fluca_CGNSViewerMeshCartesianMap", (PetscObject *)&map));
    PetscCall(FlucaMapGetValue(map, (PetscObject)sol->mesh, (PetscObject *)&viewerinfo_container));
    PetscCall(PetscContainerGetPointer(viewerinfo_container, (void **)&viewerinfo));

    PetscCallCGNS(cg_sol_write(cgns->file_num, cgns->base, viewerinfo->zone, "Solution", CGNS_ENUMV(CellCenter),
                               &viewerinfo->sol));

    PetscCall(MeshGetDM(sol->mesh, &dm));
    PetscCall(MeshGetDim(sol->mesh, &dim));
    for (d = 0; d < dim; d++)
        PetscCall(FlucaViewerCGNSWriteStructuredSolution_Private(dm, sol->v[d], cgns->file_num, cgns->base,
                                                                 viewerinfo->zone, viewerinfo->sol, velnames[d]));
    PetscCall(FlucaViewerCGNSWriteStructuredSolution_Private(dm, sol->p, cgns->file_num, cgns->base, viewerinfo->zone,
                                                             viewerinfo->sol, "Pressure"));

    PetscFunctionReturn(PETSC_SUCCESS);
}
