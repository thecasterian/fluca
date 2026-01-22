#pragma once

#include <flucasys.h>
#include <petscmat.h>
#include <petscdmstag.h>

/* FlucaFD - Finite Difference Operator */
typedef struct _p_FlucaFD *FlucaFD;

/* FlucaFD types */
typedef const char *FlucaFDType;
#define FLUCAFDDERIVATIVE    "derivative"
#define FLUCAFDINTERPOLATION "interpolation"
#define FLUCAFDCOMPOSITION   "composition"
#define FLUCAFDSCALE         "scale"
#define FLUCAFDSUM           "sum"

/* Enums */
typedef enum {
  FLUCAFD_ELEMENT,
  FLUCAFD_LEFT,
  FLUCAFD_DOWN,
  FLUCAFD_BACK,
} FlucaFDStencilLocation;
FLUCA_EXTERN const char *FlucaFDStencilLocations[];

typedef enum {
  FLUCAFD_X,
  FLUCAFD_Y,
  FLUCAFD_Z
} FlucaFDDirection;
FLUCA_EXTERN const char *FlucaFDDirections[];

typedef enum {
  FLUCAFD_BC_NONE,
  FLUCAFD_BC_DIRICHLET,
  FLUCAFD_BC_NEUMANN,
  FLUCAFD_BC_PERIODIC
} FlucaFDBoundaryConditionType;
FLUCA_EXTERN const char *FlucaFDBoundaryConditionTypes[];

typedef struct {
  FlucaFDBoundaryConditionType type;
  PetscScalar                  value; /* for Dirichlet/Neumann boundary conditions */
} FlucaFDBoundaryCondition;

FLUCA_EXTERN PetscErrorCode FlucaFDCreate(MPI_Comm, FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDSetType(FlucaFD, FlucaFDType);
FLUCA_EXTERN PetscErrorCode FlucaFDGetType(FlucaFD, FlucaFDType *);
FLUCA_EXTERN PetscErrorCode FlucaFDSetUp(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDDestroy(FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDView(FlucaFD, PetscViewer);
FLUCA_EXTERN PetscErrorCode FlucaFDViewFromOptions(FlucaFD, PetscObject, const char[]);

FLUCA_EXTERN PetscErrorCode FlucaFDSetCoordinateDM(FlucaFD, DM);
FLUCA_EXTERN PetscErrorCode FlucaFDSetInputLocation(FlucaFD, FlucaFDStencilLocation, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDSetOutputLocation(FlucaFD, FlucaFDStencilLocation, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDSetBoundaryConditions(FlucaFD, const FlucaFDBoundaryCondition[]);
FLUCA_EXTERN PetscErrorCode FlucaFDSetFromOptions(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDSetOptionsPrefix(FlucaFD, const char[]);
FLUCA_EXTERN PetscErrorCode FlucaFDAppendOptionsPrefix(FlucaFD, const char[]);
FLUCA_EXTERN PetscErrorCode FlucaFDGetOptionsPrefix(FlucaFD, const char *[]);

FLUCA_EXTERN PetscErrorCode FlucaFDGetStencilRaw(FlucaFD, PetscInt, PetscInt, PetscInt, PetscInt *, DMStagStencil[], PetscScalar[]);
FLUCA_EXTERN PetscErrorCode FlucaFDGetStencil(FlucaFD, PetscInt, PetscInt, PetscInt, PetscInt *, DMStagStencil[], PetscScalar[]);
FLUCA_EXTERN PetscErrorCode FlucaFDApply(FlucaFD, DM, DM, Mat);

/* FLUCAFDDERIVATIVE specific */
FLUCA_EXTERN PetscErrorCode FlucaFDDerivativeSetDerivativeOrder(FlucaFD, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDDerivativeSetAccuracyOrder(FlucaFD, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDDerivativeSetDirection(FlucaFD, FlucaFDDirection);

/* FLUCAFDCOMPOSITION specific */
FLUCA_EXTERN PetscErrorCode FlucaFDCompositionSetOperands(FlucaFD, FlucaFD, FlucaFD);

/* FLUCAFDSCALE specific */
FLUCA_EXTERN PetscErrorCode FlucaFDScaleSetOperand(FlucaFD, FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDScaleSetConstant(FlucaFD, PetscScalar);
FLUCA_EXTERN PetscErrorCode FlucaFDScaleSetVector(FlucaFD, Vec, FlucaFDStencilLocation, PetscInt);

/* FLUCAFDSUM specific */
FLUCA_EXTERN PetscErrorCode FlucaFDSumGetNumOperands(FlucaFD, PetscInt *);
FLUCA_EXTERN PetscErrorCode FlucaFDSumAddOperand(FlucaFD, FlucaFD);

FLUCA_EXTERN PetscErrorCode FlucaFDInitializePackage(void);
FLUCA_EXTERN PetscErrorCode FlucaFDFinalizePackage(void);

FLUCA_EXTERN PetscFunctionList FlucaFDList;
FLUCA_EXTERN PetscErrorCode    FlucaFDRegister(const char[], PetscErrorCode (*)(FlucaFD));
