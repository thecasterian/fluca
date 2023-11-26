#include <fluca/private/nsimpl.h>
#include <fluca/private/viewercgnsutils.h>
#include <flucamap.h>
#include <petsc/private/viewercgnsimpl.h>

PetscErrorCode NSView_CGNS(NS ns, PetscViewer v) {
    PetscViewer_CGNS *cgns = (PetscViewer_CGNS *)v->data;
    FlucaMap map;
    PetscContainer viewerinfo_container;
    ViewerCGNSInfo *viewerinfo;

    const int nsteps = 1;
    const cgsize_t biterdim[1] = {1};
    const int max_name_len = 32; /* Specified in SIDS. */
    const cgsize_t ptrdim[2] = {max_name_len, 1};
    const cgsize_t base_rmin[1] = {1}, base_rmax[1] = {1};
    const double time[1] = {ns->t};
    const int step[1] = {ns->step};
    const cgsize_t zone_rmin[2] = {1, 1}, zone_rmax[2] = {max_name_len, 1};
    int array;

    PetscFunctionBegin;

    if (!cgns->file_num)
        FlucaViewerCGNSFileOpen_Private(v, PETSC_TRUE, ns->step);

    PetscCall(SolView(ns->sol, v));

    PetscCall(PetscObjectQuery((PetscObject)v, "_Fluca_CGNSViewerMeshCartesianMap", (PetscObject *)&map));
    PetscCall(FlucaMapGetValue(map, (PetscObject)ns->mesh, (PetscObject *)&viewerinfo_container));
    PetscCall(PetscContainerGetPointer(viewerinfo_container, (void **)&viewerinfo));

    /* Write transient informations. */
    PetscCallCGNS(cg_biter_write(cgns->file_num, cgns->base, "TimeIterValues", nsteps));
    PetscCallCGNS(cg_goto(cgns->file_num, cgns->base, "BaseIterativeData_t", 1, NULL));
    PetscCallCGNS(cgp_array_write("TimeValues", CGNS_ENUMV(RealDouble), 1, biterdim, &array));
    PetscCallCGNS(cgp_array_write_data(array, base_rmin, base_rmax, time));
    PetscCallCGNS(cgp_array_write("IterationValues", CGNS_ENUMV(Integer), 1, biterdim, &array));
    PetscCallCGNS(cgp_array_write_data(array, base_rmin, base_rmax, step));

    PetscCallCGNS(cg_ziter_write(cgns->file_num, cgns->base, viewerinfo->zone, "ZoneIterData"));
    PetscCallCGNS(cg_goto(cgns->file_num, cgns->base, "Zone_t", viewerinfo->zone, "ZoneIterativeData_t", 1, NULL));
    PetscCallCGNS(cgp_array_write("FlowSolutionPointers", CGNS_ENUMV(Character), 2, ptrdim, &array));
    PetscCallCGNS(cgp_array_write_data(array, zone_rmin, zone_rmax, "Solution"));

    PetscFunctionReturn(PETSC_SUCCESS);
}
