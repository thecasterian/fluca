#if !defined(FLUCANS_H)
#define FLUCANS_H

#include <flucameshtypes.h>
#include <flucasys.h>

typedef struct _p_NS *NS;

typedef const char *NSType;
#define NSFSM "fsm"

FLUCA_EXTERN PetscClassId NS_CLASSID;

FLUCA_EXTERN PetscErrorCode NSInitializePackage(void);
FLUCA_EXTERN PetscErrorCode NSFinalizePackage(void);

FLUCA_EXTERN PetscErrorCode NSCreate(MPI_Comm, NS *);
FLUCA_EXTERN PetscErrorCode NSSetType(NS, NSType);
FLUCA_EXTERN PetscErrorCode NSGetType(NS, NSType *);
FLUCA_EXTERN PetscErrorCode NSSetMesh(NS, Mesh);
FLUCA_EXTERN PetscErrorCode NSGetMesh(NS, Mesh *);
FLUCA_EXTERN PetscErrorCode NSSetUp(NS);
FLUCA_EXTERN PetscErrorCode NSDestroy(NS *);
FLUCA_EXTERN PetscErrorCode NSView(NS, PetscViewer);
FLUCA_EXTERN PetscErrorCode NSViewFromOptions(NS, PetscObject, const char *);

FLUCA_EXTERN PetscFunctionList NSList;
FLUCA_EXTERN PetscErrorCode NSRegister(const char *, PetscErrorCode (*)(NS));

#endif
