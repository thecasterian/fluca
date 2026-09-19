#pragma once

#include <flucasys.h>
#include <petscts.h>
#include <petscdmstag.h>

/* Phys - Physical Model */
typedef struct _p_Phys *Phys;

/* Phys types */
typedef const char *PhysType;
#define PHYSINS "ins" /* Incompressible Navier-Stokes */

FLUCA_EXTERN PetscClassId   PHYS_CLASSID;
FLUCA_EXTERN PetscErrorCode PhysInitializePackage(void);
FLUCA_EXTERN PetscErrorCode PhysFinalizePackage(void);

/* Body force callback */
typedef PetscErrorCode PhysBodyForceFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx);

/* INS boundary condition types */
typedef enum {
  PHYS_INS_BC_NONE,
  PHYS_INS_BC_VELOCITY,
} PhysINSBCType;
FLUCA_EXTERN const char *PhysINSBCTypes[];

/* INS boundary condition callback: returns value of field component at boundary coordinates.
   comp is the solution DOF component being queried (0..dim-1 for velocity, dim for pressure). */
typedef PetscErrorCode PhysINSBCFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx);

typedef struct {
  PhysINSBCType type;
  PhysINSBCFn  *fn; /* value BC: u_bc(t, x, comp) */
  void         *ctx;
  PhysINSBCFn  *fn_dot; /* time derivative BC: du_bc/dt(t, x, comp); NULL = use FD approx of fn */
  void         *fn_dot_ctx;
} PhysINSBC;

/* Lifecycle */
FLUCA_EXTERN PetscErrorCode PhysCreate(MPI_Comm, Phys *);
FLUCA_EXTERN PetscErrorCode PhysSetType(Phys, PhysType);
FLUCA_EXTERN PetscErrorCode PhysGetType(Phys, PhysType *);
FLUCA_EXTERN PetscErrorCode PhysSetBaseDM(Phys, DM);
FLUCA_EXTERN PetscErrorCode PhysGetBaseDM(Phys, DM *);
FLUCA_EXTERN PetscErrorCode PhysGetSolutionDM(Phys, DM *);
FLUCA_EXTERN PetscErrorCode PhysSetFromOptions(Phys);
FLUCA_EXTERN PetscErrorCode PhysSetUp(Phys);
FLUCA_EXTERN PetscErrorCode PhysDestroy(Phys *);
FLUCA_EXTERN PetscErrorCode PhysView(Phys, PetscViewer);
FLUCA_EXTERN PetscErrorCode PhysViewFromOptions(Phys, PetscObject, const char[]);

/* Solution fields */
typedef enum {
  PHYS_FIELD_ELEMENT,
  PHYS_FIELD_FACE,
} PhysFieldLocation;
FLUCA_EXTERN const char *PhysFieldLocations[];

#define PHYS_FIELD_VELOCITY      "velocity"
#define PHYS_FIELD_PRESSURE      "pressure"
#define PHYS_FIELD_FACE_VELOCITY "face_velocity"

FLUCA_EXTERN PetscErrorCode PhysGetField(Phys, const char[], PhysFieldLocation *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode PhysGetFieldIS(Phys, const char[], IS *);

/* Material properties */
FLUCA_EXTERN PetscErrorCode PhysGetDensity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysGetViscosity(Phys, PetscReal *);

/* Rows of the coupled system (13) of the theory guide, assembled on the solution DM */
FLUCA_EXTERN PetscErrorCode PhysComputeMomentumSystem(Phys, PetscReal, PetscReal, Vec, Mat, Vec);
FLUCA_EXTERN PetscErrorCode PhysComputeCouplingSystem(Phys, PetscReal, PetscReal, Mat, Vec);

/* Options prefix */
FLUCA_EXTERN PetscErrorCode PhysSetOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysAppendOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysGetOptionsPrefix(Phys, const char *[]);

/* Body force (base class) */
FLUCA_EXTERN PetscErrorCode PhysSetBodyForce(Phys, PhysBodyForceFn *, void *);

/* PHYSINS specific */
FLUCA_EXTERN PetscErrorCode PhysINSSetDensity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysINSGetDensity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysINSSetViscosity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysINSGetViscosity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysINSSetBoundaryCondition(Phys, PetscInt, PhysINSBC);
FLUCA_EXTERN PetscErrorCode PhysINSGetBoundaryCondition(Phys, PetscInt, PhysINSBC *);

/* Registration */
FLUCA_EXTERN PetscFunctionList PhysList;
FLUCA_EXTERN PetscErrorCode    PhysRegister(const char[], PetscErrorCode (*)(Phys));

/* Approximate block factorization preconditioner for the coupled system (13) */
#define PCABF "abf"

FLUCA_EXTERN PetscErrorCode PCABFSetFields(PC, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode PCABFGetSubKSPs(PC, KSP *, KSP *);

typedef enum {
  PC_ABF_AINV_ID,
  PC_ABF_AINV_DIAG,
  PC_ABF_AINV_ROWSUM,
} PCABFAinvType;
FLUCA_EXTERN const char *const PCABFAinvTypes[];

FLUCA_EXTERN PetscErrorCode PCABFSetSchurComplementAinvType(PC, PCABFAinvType);
FLUCA_EXTERN PetscErrorCode PCABFSetUpperTriangularAinvType(PC, PCABFAinvType);
