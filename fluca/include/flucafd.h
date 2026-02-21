#pragma once

#include <flucasys.h>
#include <petscmat.h>
#include <petscdmstag.h>

/* FlucaFD - Finite Difference Operator */
typedef struct _p_FlucaFD *FlucaFD;

/* FlucaFD types */
typedef const char *FlucaFDType;
#define FLUCAFDDERIVATIVE     "derivative"
#define FLUCAFDINTERPOLATION  "interpolation"
#define FLUCAFDCOMPOSITION    "composition"
#define FLUCAFDSCALE          "scale"
#define FLUCAFDSUM            "sum"
#define FLUCAFDSECONDORDERTVD "secondordertvd"

FLUCA_EXTERN PetscClassId   FLUCAFD_CLASSID;
FLUCA_EXTERN PetscErrorCode FlucaFDInitializePackage(void);
FLUCA_EXTERN PetscErrorCode FlucaFDFinalizePackage(void);

/* Enums */
typedef enum {
  FLUCAFD_X,
  FLUCAFD_Y,
  FLUCAFD_Z,
} FlucaFDDirection;
FLUCA_EXTERN const char *FlucaFDDirections[];

typedef enum {
  FLUCAFD_BC_NONE,
  FLUCAFD_BC_DIRICHLET,
  FLUCAFD_BC_NEUMANN,
} FlucaFDBoundaryConditionType;
FLUCA_EXTERN const char *FlucaFDBoundaryConditionTypes[];

typedef struct {
  FlucaFDBoundaryConditionType type;
  PetscScalar                  value; /* for Dirichlet/Neumann boundary conditions */
} FlucaFDBoundaryCondition;

/* Boundary value component markers for stencil points */
#define FLUCAFD_BOUNDARY_LEFT  (-1)
#define FLUCAFD_BOUNDARY_RIGHT (-2)
#define FLUCAFD_BOUNDARY_DOWN  (-3)
#define FLUCAFD_BOUNDARY_UP    (-4)
#define FLUCAFD_BOUNDARY_BACK  (-5)
#define FLUCAFD_BOUNDARY_FRONT (-6)
/* Constant term marker for stencil points */
#define FLUCAFD_CONSTANT (-7)

FLUCA_EXTERN PetscErrorCode FlucaFDCreate(MPI_Comm, FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDSetType(FlucaFD, FlucaFDType);
FLUCA_EXTERN PetscErrorCode FlucaFDGetType(FlucaFD, FlucaFDType *);
FLUCA_EXTERN PetscErrorCode FlucaFDSetUp(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDDestroy(FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDView(FlucaFD, PetscViewer);
FLUCA_EXTERN PetscErrorCode FlucaFDViewFromOptions(FlucaFD, PetscObject, const char[]);

FLUCA_EXTERN PetscErrorCode FlucaFDSetDM(FlucaFD, DM);
FLUCA_EXTERN PetscErrorCode FlucaFDSetInputLocation(FlucaFD, DMStagStencilLocation, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDSetOutputLocation(FlucaFD, DMStagStencilLocation, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDSetBoundaryConditions(FlucaFD, const FlucaFDBoundaryCondition[]);
FLUCA_EXTERN PetscErrorCode FlucaFDGetBoundaryConditions(FlucaFD, FlucaFDBoundaryCondition[]);
FLUCA_EXTERN PetscErrorCode FlucaFDSetFromOptions(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDSetOptionsPrefix(FlucaFD, const char[]);
FLUCA_EXTERN PetscErrorCode FlucaFDAppendOptionsPrefix(FlucaFD, const char[]);
FLUCA_EXTERN PetscErrorCode FlucaFDGetOptionsPrefix(FlucaFD, const char *[]);

FLUCA_EXTERN PetscErrorCode FlucaFDGetStencilRaw(FlucaFD, PetscInt, PetscInt, PetscInt, PetscInt *, DMStagStencil[], PetscScalar[]);
FLUCA_EXTERN PetscErrorCode FlucaFDGetStencil(FlucaFD, PetscInt, PetscInt, PetscInt, PetscInt *, DMStagStencil[], PetscScalar[]);
FLUCA_EXTERN PetscErrorCode FlucaFDApply(FlucaFD, DM, DM, Vec, Vec);
FLUCA_EXTERN PetscErrorCode FlucaFDGetOperator(FlucaFD, DM, DM, Mat);

/* FLUCAFDDERIVATIVE specific */
FLUCA_EXTERN PetscErrorCode FlucaFDDerivativeCreate(DM, FlucaFDDirection, PetscInt, PetscInt, DMStagStencilLocation, PetscInt, DMStagStencilLocation, PetscInt, FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDDerivativeSetDerivativeOrder(FlucaFD, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDDerivativeSetAccuracyOrder(FlucaFD, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDDerivativeSetDirection(FlucaFD, FlucaFDDirection);

/* FLUCAFDCOMPOSITION specific */
FLUCA_EXTERN PetscErrorCode FlucaFDCompositionCreate(FlucaFD, FlucaFD, FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDCompositionSetOperands(FlucaFD, FlucaFD, FlucaFD);

/* FLUCAFDSCALE specific */
FLUCA_EXTERN PetscErrorCode FlucaFDScaleCreateConstant(FlucaFD, PetscScalar, FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDScaleCreateVector(FlucaFD, Vec, PetscInt, FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDScaleSetOperand(FlucaFD, FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDScaleSetConstant(FlucaFD, PetscScalar);
FLUCA_EXTERN PetscErrorCode FlucaFDScaleSetVector(FlucaFD, Vec, DMStagStencilLocation, PetscInt);

/* FLUCAFDSUM specific */
FLUCA_EXTERN PetscErrorCode FlucaFDSumCreate(PetscInt, const FlucaFD[], FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDSumGetNumOperands(FlucaFD, PetscInt *);
FLUCA_EXTERN PetscErrorCode FlucaFDSumAddOperand(FlucaFD, FlucaFD);

/* FLUCAFDSECONDORDERTVD specific */
typedef PetscScalar            FlucaFDLimiterFn(PetscScalar);
FLUCA_EXTERN PetscFunctionList FlucaFDLimiterList;

FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDCreate(DM, FlucaFDDirection, PetscInt, PetscInt, FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetDirection(FlucaFD, FlucaFDDirection);
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetLimiter(FlucaFD, const char *);
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetVelocity(FlucaFD, Vec, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetCurrentSolution(FlucaFD, Vec);

FLUCA_EXTERN PetscFunctionList FlucaFDList;
FLUCA_EXTERN PetscErrorCode    FlucaFDRegister(const char[], PetscErrorCode (*)(FlucaFD));
