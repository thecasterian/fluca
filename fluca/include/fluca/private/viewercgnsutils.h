#if !defined(VIEWERCGNSUTILS_H)
#define VIEWERCGNSUTILS_H

#include <fluca/private/flucaimpl.h>
#include <pcgnslib.h>
#include <petscdmda.h>
#include <petscvec.h>

typedef struct {
    int zone;
    int coord[3];
    int sol;
} ViewerCGNSInfo;

FLUCA_EXTERN PetscErrorCode FlucaViewerCGNSFileOpen_Private(PetscViewer v, int sequence_number);
FLUCA_EXTERN PetscErrorCode FlucaViewerCGNSWriteStructuredSolution_Private(DM da, Vec v, int file_num, int base,
                                                                           int zone, int sol, const char *name);

#endif
