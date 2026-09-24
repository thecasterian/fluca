#pragma once

#include <flucasys.h>
#include <petscdmstag.h>

/* Phys - Physical Model */
typedef struct _p_Phys *Phys;

/* Phys types */
typedef const char *PhysType;
#define PHYSLAMINAR "laminar" /* Isothermal laminar incompressible flow */

FLUCA_EXTERN PetscClassId   PHYS_CLASSID;
FLUCA_EXTERN PetscErrorCode PhysInitializePackage(void);
FLUCA_EXTERN PetscErrorCode PhysFinalizePackage(void);

/* Body force callback */
typedef PetscErrorCode PhysBodyForceFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscScalar f[], void *ctx);

/* Boundary conditions */
typedef enum {
  PHYS_BC_NONE,
  PHYS_BC_VELOCITY,
} PhysBCType;
FLUCA_EXTERN const char *PhysBCTypes[];

/* Boundary condition callback: returns value of field component at boundary coordinates.
   comp is the solution DOF component being queried (0..dim-1 for velocity, dim for pressure). */
typedef PetscErrorCode PhysBCFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx);

typedef struct {
  PhysBCType type;
  PhysBCFn  *fn; /* value BC: u_bc(t, x, comp) */
  void      *ctx;
  PhysBCFn  *fn_dot; /* time derivative BC: du_bc/dt(t, x, comp); NULL = use FD approx of fn */
  void      *fn_dot_ctx;
} PhysBC;

FLUCA_EXTERN PetscErrorCode PhysSetBoundaryCondition(Phys, PetscInt, PhysBC);
FLUCA_EXTERN PetscErrorCode PhysGetBoundaryCondition(Phys, PetscInt, PhysBC *);

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

/* What equation a field satisfies. Seg builds operators from this. */
typedef enum {
  PHYS_EQN_MOMENTUM,           /* rho Du/Dt = -grad p + div(mu grad u) */
  PHYS_EQN_PRESSURE,           /* the incompressibility constraint */
  PHYS_EQN_TRANSPORTED_SCALAR, /* reserved; rejected by Seg */
  PHYS_EQN_AUXILIARY,          /* satisfies no PDE of its own; the Seg defines its rows */
} PhysEquationRole;
FLUCA_EXTERN const char *PhysEquationRoles[];

FLUCA_EXTERN PetscErrorCode PhysGetFieldRole(Phys, const char[], PhysEquationRole *);
FLUCA_EXTERN PetscErrorCode PhysGetFieldNullSpaceConstant(Phys, const char[], PetscBool *);
FLUCA_EXTERN PetscErrorCode PhysGetNumFields(Phys, PetscInt *);
FLUCA_EXTERN PetscErrorCode PhysGetFieldName(Phys, PetscInt, const char *[]);

#define PHYS_FIELD_VELOCITY      "velocity"
#define PHYS_FIELD_PRESSURE      "pressure"
#define PHYS_FIELD_FACE_VELOCITY "face_velocity"

FLUCA_EXTERN PetscErrorCode PhysGetField(Phys, const char[], PhysFieldLocation *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode PhysGetFieldIS(Phys, const char[], IS *);

/* Material properties.

   Only PHYS_PROPERTY_CONSTANT is accepted. FUNCTION and FIELD are reserved: the enum, the
   registration call and the queries carry them so that supporting them later does not change
   any signature, but registering one raises PETSC_ERR_SUP. */
typedef enum {
  PHYS_PROPERTY_CONSTANT,
  PHYS_PROPERTY_FUNCTION,
  PHYS_PROPERTY_FIELD,
} PhysPropertySource;
FLUCA_EXTERN const char *PhysPropertySources[];

#define PHYS_PROPERTY_DENSITY   "density"
#define PHYS_PROPERTY_VISCOSITY "viscosity"

FLUCA_EXTERN PetscErrorCode PhysSetDensity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysGetDensity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysSetViscosity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysGetViscosity(Phys, PetscReal *);

FLUCA_EXTERN PetscErrorCode PhysGetPropertySource(Phys, const char[], PhysPropertySource *);
FLUCA_EXTERN PetscErrorCode PhysGetPropertyConstant(Phys, const char[], PetscScalar *);
FLUCA_EXTERN PetscErrorCode PhysGetPropertyLocation(Phys, const char[], PhysFieldLocation *);

/* Options prefix */
FLUCA_EXTERN PetscErrorCode PhysSetOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysAppendOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysGetOptionsPrefix(Phys, const char *[]);

/* Body force (base class) */
FLUCA_EXTERN PetscErrorCode PhysSetBodyForce(Phys, PhysBodyForceFn *, void *);
FLUCA_EXTERN PetscErrorCode PhysGetBodyForce(Phys, PhysBodyForceFn **, void **);

/* Registration */
FLUCA_EXTERN PetscFunctionList PhysList;
FLUCA_EXTERN PetscErrorCode    PhysRegister(const char[], PetscErrorCode (*)(Phys));
