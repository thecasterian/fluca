#if !defined(VIEWERCGNSUTILS_H)
#define VIEWERCGNSUTILS_H

#include <pcgnslib.h>
#include <petscdmda.h>
#include <petscvec.h>

typedef struct {
    int zone;
    int coord[3];
    int sol;
} ViewerCGNSInfo;

FLUCA_EXTERN PetscErrorCode ViewerCGNSWriteStructuredSolution_Private(DM da, Vec v, int file_num, int base, int zone,
                                                                      int sol, const char *name);

#endif
