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

FLUCA_EXTERN PetscErrorCode FlucaViewerCGNSFileOpen_Private(PetscViewer, PetscBool, int);
FLUCA_EXTERN PetscErrorCode FlucaViewerCGNSWriteStructuredSolution_Private(DM, Vec, int, int, int, int, const char *);

#endif
