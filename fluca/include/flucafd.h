#pragma once

#include <flucasys.h>
#include <petscmat.h>
#include <petscdmstag.h>

/* FlucaFD - Finite Difference Operator */
typedef struct _p_FlucaFD *FlucaFD;

/* FlucaFD types */
typedef const char *FlucaFDType;
#define FLUCAFDDERIVATIVE     "derivative"
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

/* Callback for spatially varying boundary conditions.
   comp is the source DOF component of the off-grid stencil point. */
typedef PetscErrorCode FlucaFDBoundaryConditionFn(PetscInt dim, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx);

typedef struct {
  FlucaFDBoundaryConditionType type;
  PetscScalar                  value; /* uniform value (used when fn == NULL) */
  FlucaFDBoundaryConditionFn  *fn;    /* spatially varying callback (takes priority over value) */
  void                        *ctx;   /* callback context */
} FlucaFDBoundaryCondition;

/* Boundary marker encoding: packs face index and source component into col[c].c.
   Face index ∈ [0, NFACES-1], comp ≥ 0.
   For single-component operators (comp=0), markers are -1..-6 (backward compatible). */
#define FLUCAFD_NFACES                      6 /* 2 faces per dimension, max 3 dimensions */
#define FLUCAFD_BOUNDARY_MARKER(face, comp) (-(1 + (face) + FLUCAFD_NFACES * (comp)))
#define FLUCAFD_BOUNDARY_FACE(marker)       ((-(marker)-1) % FLUCAFD_NFACES)
#define FLUCAFD_BOUNDARY_COMP(marker)       ((-(marker)-1) / FLUCAFD_NFACES)

#define FLUCAFD_BOUNDARY_LEFT  FLUCAFD_BOUNDARY_MARKER(0, 0) /* -1 */
#define FLUCAFD_BOUNDARY_RIGHT FLUCAFD_BOUNDARY_MARKER(1, 0) /* -2 */
#define FLUCAFD_BOUNDARY_DOWN  FLUCAFD_BOUNDARY_MARKER(2, 0) /* -3 */
#define FLUCAFD_BOUNDARY_UP    FLUCAFD_BOUNDARY_MARKER(3, 0) /* -4 */
#define FLUCAFD_BOUNDARY_BACK  FLUCAFD_BOUNDARY_MARKER(4, 0) /* -5 */
#define FLUCAFD_BOUNDARY_FRONT FLUCAFD_BOUNDARY_MARKER(5, 0) /* -6 */

/* Constant term marker for stencil points (must not collide with boundary markers) */
#define FLUCAFD_CONSTANT PETSC_MIN_INT

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
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetMassFlux(FlucaFD, Vec, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetCurrentSolution(FlucaFD, Vec);

FLUCA_EXTERN PetscFunctionList FlucaFDList;
FLUCA_EXTERN PetscErrorCode    FlucaFDRegister(const char[], PetscErrorCode (*)(FlucaFD));
