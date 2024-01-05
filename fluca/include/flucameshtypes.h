#if !defined(FLUCAMESHTYPES_H)
#define FLUCAMESHTYPES_H

#include <flucasys.h>

typedef struct _p_Mesh *Mesh;

typedef const char *MeshType;
#define MESHCART "cart"

typedef enum {
    MESH_BOUNDARY_NONE,
    MESH_BOUNDARY_PERIODIC,
} MeshBoundaryType;
FLUCA_EXTERN const char *MeshBoundaryTypes[];

#endif
