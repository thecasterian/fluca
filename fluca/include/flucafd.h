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

typedef PetscErrorCode (*FlucaFDBCValueFn)(PetscInt dim, const PetscReal x[], void *ctx, PetscScalar *value);

typedef struct {
  FlucaFDBoundaryConditionType type;
  PetscScalar                  value;  /* constant value (used when fn == NULL) */
  FlucaFDBCValueFn             fn;     /* function-based value (NULL = use constant) */
  void                        *fn_ctx; /* user context for fn */
} FlucaFDBoundaryCondition;

/* Stencil point types */
typedef enum {
  FLUCAFD_STENCIL_GRID,
  FLUCAFD_STENCIL_BOUNDARY,
  FLUCAFD_STENCIL_CONSTANT,
} FlucaFDStencilPointType;

/* Deferred vector scale reference. Array pointers are borrowed from FlucaFD_Scale's
   DMDA local vector and are valid only while the owning FlucaFD object is alive.
   Stencil points with nscales > 0 must not be stored beyond the current call frame. */
typedef struct {
  PetscInt    dim;
  PetscInt    i, j, k; /* position to evaluate scale vec */
  const void *arr;
} FlucaFDScaleRef;

#define FLUCAFD_MAX_SCALES 4
#define FLUCAFD_MAX_TVDS   1

/* Limiter function type (forward declaration for FlucaFDTVDRef) */
typedef PetscScalar FlucaFDLimiterFn(PetscScalar);

/* Role of a stencil point in a TVD pair */
typedef enum {
  FLUCAFD_TVD_PREV, /* cell at index-1 in the TVD direction */
  FLUCAFD_TVD_NEXT, /* cell at index   in the TVD direction */
} FlucaFDTVDRefRole;

/* Deferred TVD non-linear term reference. The owning FlucaFD must be alive
   for the internal arrays to remain valid. Stencil points with ntvds > 0
   must not be stored beyond the current call frame. */
typedef struct {
  FlucaFDTVDRefRole role;    /* PREV or NEXT */
  PetscInt          i, j, k; /* face position where TVD is evaluated */
  FlucaFDDirection  dir;
  PetscInt          N_dir;        /* domain size in TVD direction */
  PetscBool         periodic_dir; /* periodicity in TVD direction */
  FlucaFDLimiterFn *limiter;
  PetscScalar      *alpha_plus;
  PetscScalar      *alpha_minus;
  const void       *arr_mf;
  const void       *arr_phi;
  FlucaFD           fd_grad; /* gradient operator (element -> face) */
} FlucaFDTVDRef;

typedef struct {
  FlucaFDStencilPointType type;
  DMStagStencilLocation   loc;
  PetscInt                i, j, k;
  PetscInt                c;
  PetscInt                boundary_face; /* 0=left,1=right,2=down,3=up,4=back,5=front */
  PetscScalar             v;
  PetscInt                nscales;
  FlucaFDScaleRef         scales[FLUCAFD_MAX_SCALES];
  PetscInt                ntvds;
  FlucaFDTVDRef           tvds[FLUCAFD_MAX_TVDS];
} FlucaFDStencilPoint;

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
FLUCA_EXTERN PetscErrorCode FlucaFDSetBoundaryConditions(FlucaFD, PetscInt, const FlucaFDBoundaryCondition[]);
FLUCA_EXTERN PetscErrorCode FlucaFDGetBoundaryConditions(FlucaFD, PetscInt, FlucaFDBoundaryCondition[]);
FLUCA_EXTERN PetscErrorCode FlucaFDSetFromOptions(FlucaFD);
FLUCA_EXTERN PetscErrorCode FlucaFDSetOptionsPrefix(FlucaFD, const char[]);
FLUCA_EXTERN PetscErrorCode FlucaFDAppendOptionsPrefix(FlucaFD, const char[]);
FLUCA_EXTERN PetscErrorCode FlucaFDGetOptionsPrefix(FlucaFD, const char *[]);

FLUCA_EXTERN PetscErrorCode FlucaFDGetStencilRaw(FlucaFD, PetscInt, PetscInt, PetscInt, PetscInt *, FlucaFDStencilPoint[]);
FLUCA_EXTERN PetscErrorCode FlucaFDGetStencil(FlucaFD, PetscInt, PetscInt, PetscInt, PetscInt *, FlucaFDStencilPoint[]);
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
FLUCA_EXTERN PetscFunctionList FlucaFDLimiterList;

FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDCreate(DM, FlucaFDDirection, PetscInt, PetscInt, FlucaFD *);
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetDirection(FlucaFD, FlucaFDDirection);
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetLimiter(FlucaFD, const char *);
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetMassFlux(FlucaFD, Vec, PetscInt);
FLUCA_EXTERN PetscErrorCode FlucaFDSecondOrderTVDSetCurrentSolution(FlucaFD, Vec);

FLUCA_EXTERN PetscFunctionList FlucaFDList;
FLUCA_EXTERN PetscErrorCode    FlucaFDRegister(const char[], PetscErrorCode (*)(FlucaFD));
