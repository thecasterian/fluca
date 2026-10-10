#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucamesh.h>
#include <fluca/private/ibbodyimpl.h>
#include <petsc/private/hashmapi.h>

#define MESH_MIN_DIM 1
#define MESH_MAX_DIM 3

FLUCA_EXTERN PetscBool      MeshRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode MeshRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  MESH_SetUp;

#define MESH_IB_MAX_LOC    4   /* element and up to three face locations */
#define MESH_IB_MAX_BODIES 127 /* body ids are stored as int8_t */

/* Crossing of the segment from a forcing point to its solid star neighbour.
   theta is the linear root of phi along the segment; on curved walls x_ib is O(h^2) off
   the surface, which limits the forcing-row residual to O(h^2) for both interpolation orders. */
typedef struct {
  PetscInt  dir;   /* 0, 1, 2 */
  PetscInt  side;  /* 0: neighbour at index - 1, 1: at index + 1 */
  PetscReal theta; /* crossing at x_f + theta * (x_n - x_f); linear root of phi */
  PetscInt  body;  /* body id of the solid neighbour */
  PetscReal x_ib[3];
} MeshIBCut;

typedef struct {
  PetscInt  ncut;
  MeshIBCut cuts[2 * MESH_MAX_DIM];
  PetscReal normal[3]; /* unit grad(phi) at the forcing point, pointing into the fluid */
} MeshIBForcingPoint;

typedef struct {
  DMStagStencilLocation loc;
  PetscInt              slot; /* slot of loc in the dm_phi local layout */
  PetscInt              nforcing, maxforcing;
  MeshIBForcingPoint   *forcing;     /* owned forcing points */
  PetscHMapI            forcing_map; /* dm_phi local index -> entry of forcing */
} MeshIBLocation;

typedef struct {
  /* Bodies, added before MeshSetUp() */
  PetscInt nbodies;
  IBBody  *bodies; /* referenced */

  /* Built by MeshSetUp() when nbodies > 0 */
  DM                  dm_phi;                                  /* compatible with mesh->dm, one dof per classified location */
  Vec                 phi_local;                               /* every ghosted entry sampled directly, no communication */
  Vec                 phi_global;                              /* named "LevelSet" */
  const PetscScalar  *arr_phi;                                 /* read-only array of phi_local, held until destroy */
  int8_t             *body_id;                                 /* argmin body per entry of phi_local */
  PetscInt            gx[MESH_MAX_DIM], gn[MESH_MAX_DIM], epe; /* dm_phi ghost corners (0 and 1 beyond dim), entries per element */
  PetscInt            N[MESH_MAX_DIM];
  PetscBool           periodic[MESH_MAX_DIM];
  const PetscScalar **arr_coord[MESH_MAX_DIM]; /* product coordinates of mesh->dm, held until destroy */
  PetscInt            slot_coord_prev, slot_coord_elem;
  PetscInt            nloc;
  MeshIBLocation      locs[MESH_IB_MAX_LOC];
} MeshIB;

/* Whether loc lies on the faces normal to direction d */
static inline PetscBool MeshIBIsFaceInDir_Private(DMStagStencilLocation loc, PetscInt d)
{
  return (PetscBool)((d == 0 && loc == DMSTAG_LEFT) || (d == 1 && loc == DMSTAG_DOWN) || (d == 2 && loc == DMSTAG_BACK));
}

typedef struct _MeshOps *MeshOps;

struct _MeshOps {
  PetscErrorCode (*setfromoptions)(Mesh, PetscOptionItems);
  PetscErrorCode (*setup)(Mesh); /* validates mesh->dm */
  PetscErrorCode (*destroy)(Mesh);
  PetscErrorCode (*view)(Mesh, PetscViewer);
  PetscErrorCode (*load)(Mesh, PetscViewer); /* replaces mesh->dm */
};

struct _p_Mesh {
  PETSCHEADER(struct _MeshOps);

  /* Data ----------------------------------------------------------------- */
  DM       dm;   /* user-provided or loaded DMStag; referenced */
  PetscInt dim;  /* spatial dimension, from dm */
  void    *data; /* implementation-specific data */
  MeshIB   ib;   /* immersed boundary data */

  /* Status --------------------------------------------------------------- */
  PetscBool setupcalled; /* after MeshSetUp() the DM can no longer change */
};

/* MESHCARTESIAN CGNS I/O, defined in impls/cartesian/cartesiancgns.c */
FLUCA_INTERN PetscErrorCode MeshView_Cartesian_CGNS(Mesh, PetscViewer);
FLUCA_INTERN PetscErrorCode MeshLoad_Cartesian_CGNS(Mesh, PetscViewer);

/* Immersed boundary, defined in interface/meshib.c */
FLUCA_INTERN PetscErrorCode MeshIBGetLocalIndex_Internal(Mesh, const MeshIBLocation *, PetscInt, PetscInt, PetscInt, PetscInt *);
FLUCA_INTERN PetscErrorCode MeshIBClassify_Internal(Mesh, const MeshIBLocation *, PetscInt, PetscInt, PetscInt, MeshIBPointType *);
FLUCA_INTERN PetscErrorCode MeshIBDestroy_Internal(Mesh);
FLUCA_EXTERN PetscErrorCode MeshIBIsOnGrid_Internal(Mesh, DMStagStencilLocation, PetscInt, PetscInt, PetscBool *);
FLUCA_EXTERN PetscErrorCode MeshIBGetLevelSet_Internal(Mesh, DMStagStencilLocation, PetscInt, PetscInt, PetscInt, PetscReal *);
FLUCA_EXTERN PetscErrorCode MeshIBGetBodyId_Internal(Mesh, DMStagStencilLocation, PetscInt, PetscInt, PetscInt, PetscInt *);
FLUCA_EXTERN PetscErrorCode MeshIBGetPointCoordinate_Internal(Mesh, DMStagStencilLocation, PetscInt, PetscInt, PetscInt, PetscReal[]);
FLUCA_EXTERN PetscErrorCode MeshIBGetForcingPoint_Internal(Mesh, DMStagStencilLocation, PetscInt, PetscInt, PetscInt, const MeshIBForcingPoint **);
FLUCA_EXTERN PetscErrorCode MeshIBEvaluateBodyVelocity_Internal(Mesh, PetscInt, PetscReal, const PetscReal[], PetscBool, PetscScalar[]);

/* MESHCARTESIAN immersed boundary setup, defined in impls/cartesian/cartesianib.c */
FLUCA_INTERN PetscErrorCode MeshSetUpIB_Cartesian(Mesh);
