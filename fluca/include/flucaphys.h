#pragma once

#include <flucasys.h>
#include <flucamesh.h>
#include <petscdmstag.h>

/* Phys - Statement of the continuous problem: fields, boundary conditions and material properties */
typedef struct _p_Phys *Phys;

/* Phys types */
typedef const char *PhysType;
#define PHYSLAMINAR "laminar" /* Isothermal laminar incompressible flow */

/* Limits. The face index of PhysSet/GetBoundaryCondition() ranges over [0, PHYS_MAX_FACES):
   left, right, down, up, back, front. */
#define PHYS_MAX_DIM   3
#define PHYS_MAX_FACES (2 * PHYS_MAX_DIM)

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

/* Boundary condition callback: value of velocity component comp at boundary point x and time t */
typedef PetscErrorCode PhysBCFn(PetscInt dim, PetscReal t, const PetscReal x[], PetscInt comp, PetscScalar *val, void *ctx);

typedef struct {
  PhysBCType type;
  PhysBCFn  *fn; /* value; NULL means zero */
  void      *ctx;
  PhysBCFn  *fn_dot; /* time derivative; NULL means a finite difference of fn */
  void      *fn_dot_ctx;
} PhysBC;

/* Lifecycle */
FLUCA_EXTERN PetscErrorCode PhysCreate(MPI_Comm, Phys *);
FLUCA_EXTERN PetscErrorCode PhysSetType(Phys, PhysType);
FLUCA_EXTERN PetscErrorCode PhysGetType(Phys, PhysType *);
FLUCA_EXTERN PetscErrorCode PhysSetMesh(Phys, Mesh);
FLUCA_EXTERN PetscErrorCode PhysGetMesh(Phys, Mesh *);
FLUCA_EXTERN PetscErrorCode PhysGetSolutionDM(Phys, DM *);
FLUCA_EXTERN PetscErrorCode PhysSetFromOptions(Phys);
FLUCA_EXTERN PetscErrorCode PhysSetUp(Phys);
FLUCA_EXTERN PetscErrorCode PhysDestroy(Phys *);
FLUCA_EXTERN PetscErrorCode PhysView(Phys, PetscViewer);
FLUCA_EXTERN PetscErrorCode PhysViewFromOptions(Phys, PetscObject, const char[]);

/* Options prefix */
FLUCA_EXTERN PetscErrorCode PhysSetOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysAppendOptionsPrefix(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysGetOptionsPrefix(Phys, const char *[]);

/* Boundary conditions */
FLUCA_EXTERN PetscErrorCode PhysSetBoundaryCondition(Phys, PetscInt, PhysBC);
FLUCA_EXTERN PetscErrorCode PhysGetBoundaryCondition(Phys, PetscInt, PhysBC *);

/* Material properties (all constant) */
#define PHYS_PROPERTY_DENSITY   "density"
#define PHYS_PROPERTY_VISCOSITY "viscosity"

FLUCA_EXTERN PetscErrorCode PhysSetDensity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysGetDensity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysSetViscosity(Phys, PetscReal);
FLUCA_EXTERN PetscErrorCode PhysGetViscosity(Phys, PetscReal *);
FLUCA_EXTERN PetscErrorCode PhysGetProperty(Phys, const char[], PetscScalar *);

/* Body force */
FLUCA_EXTERN PetscErrorCode PhysSetBodyForce(Phys, PhysBodyForceFn *, void *);
FLUCA_EXTERN PetscErrorCode PhysGetBodyForce(Phys, PhysBodyForceFn **, void **);

/* Solution fields. PhysSetUp() declares them and lays out one DMStag holding all of them. */
typedef enum {
  PHYS_FIELD_ELEMENT,
  PHYS_FIELD_FACE,
} PhysFieldLocation;
FLUCA_EXTERN const char *PhysFieldLocations[];

/* Field names are CGNS data-name identifiers (SIDS Appendix A), since solution vectors are written to CGNS under them; a
   field with 2 or 3 components is written as <name>X, <name>Y, <name>Z. Declare custom fields the same way, e.g.
   "Temperature". */
#define PHYS_FIELD_VELOCITY      "Velocity"
#define PHYS_FIELD_FACE_VELOCITY "VelocityNormal" /* velocity normal to each face, q.n */
#define PHYS_FIELD_PRESSURE      "Pressure"

FLUCA_EXTERN PetscErrorCode PhysDeclareField(Phys, const char[], PhysFieldLocation, PetscInt);
FLUCA_EXTERN PetscErrorCode PhysRemoveField(Phys, const char[]);
FLUCA_EXTERN PetscErrorCode PhysResetFields(Phys);
FLUCA_EXTERN PetscErrorCode PhysGetNumFields(Phys, PetscInt *);
FLUCA_EXTERN PetscErrorCode PhysGetFieldName(Phys, PetscInt, const char *[]);
FLUCA_EXTERN PetscErrorCode PhysGetField(Phys, const char[], PhysFieldLocation *, PetscInt *, PetscInt *);
FLUCA_EXTERN PetscErrorCode PhysGetFieldIS(Phys, const char[], IS *);
FLUCA_EXTERN PetscErrorCode PhysCreateSolutionVector(Phys, Vec *);

/* Registration */
FLUCA_EXTERN PetscFunctionList PhysList;
FLUCA_EXTERN PetscErrorCode    PhysRegister(const char[], PetscErrorCode (*)(Phys));
