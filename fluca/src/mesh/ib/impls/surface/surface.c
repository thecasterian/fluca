#include <fluca/private/ibbodyimpl.h>

#define IBBODY_SURFACE_BAND 3. /* narrow band half-width in units of the largest grid spacing */

static PetscReal Dot3_Private(const PetscReal a[], const PetscReal b[])
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static PetscReal DistancePointSegment_Private(const PetscReal p[], const PetscReal a[], const PetscReal b[])
{
  PetscReal ab[2], t, l2, dx, dy;

  ab[0] = b[0] - a[0];
  ab[1] = b[1] - a[1];
  l2    = ab[0] * ab[0] + ab[1] * ab[1];
  t     = l2 > 0. ? ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2 : 0.;
  t     = PetscMin(PetscMax(t, 0.), 1.);
  dx    = p[0] - (a[0] + t * ab[0]);
  dy    = p[1] - (a[1] + t * ab[1]);
  return PetscSqrtReal(dx * dx + dy * dy);
}

/* Closest point on a triangle (Ericson, Real-Time Collision Detection, 5.1.5) */
static PetscReal DistancePointTriangle_Private(const PetscReal p[], const PetscReal a[], const PetscReal b[], const PetscReal c[])
{
  PetscReal ab[3], ac[3], ap[3], bp[3], cp[3], n[3], q[3], r[3], d1, d2, d3, d4, d5, d6, va, vb, vc, v, w, denom;
  PetscInt  d;

  for (d = 0; d < 3; ++d) {
    ab[d] = b[d] - a[d];
    ac[d] = c[d] - a[d];
    ap[d] = p[d] - a[d];
    bp[d] = p[d] - b[d];
    cp[d] = p[d] - c[d];
  }
  /* A zero-area triangle would divide by zero below; its edges belong to neighboring triangles */
  n[0] = ab[1] * ac[2] - ab[2] * ac[1];
  n[1] = ab[2] * ac[0] - ab[0] * ac[2];
  n[2] = ab[0] * ac[1] - ab[1] * ac[0];
  if (Dot3_Private(n, n) == 0.) return PETSC_MAX_REAL;
  d1 = Dot3_Private(ab, ap);
  d2 = Dot3_Private(ac, ap);
  d3 = Dot3_Private(ab, bp);
  d4 = Dot3_Private(ac, bp);
  d5 = Dot3_Private(ab, cp);
  d6 = Dot3_Private(ac, cp);
  vc = d1 * d4 - d3 * d2;
  vb = d5 * d2 - d1 * d6;
  va = d3 * d6 - d5 * d4;
  if (d1 <= 0. && d2 <= 0.) {
    for (d = 0; d < 3; ++d) q[d] = a[d];
  } else if (d3 >= 0. && d4 <= d3) {
    for (d = 0; d < 3; ++d) q[d] = b[d];
  } else if (vc <= 0. && d1 >= 0. && d3 <= 0.) {
    v = d1 / (d1 - d3);
    for (d = 0; d < 3; ++d) q[d] = a[d] + v * ab[d];
  } else if (d6 >= 0. && d5 <= d6) {
    for (d = 0; d < 3; ++d) q[d] = c[d];
  } else if (vb <= 0. && d2 >= 0. && d6 <= 0.) {
    w = d2 / (d2 - d6);
    for (d = 0; d < 3; ++d) q[d] = a[d] + w * ac[d];
  } else if (va <= 0. && d4 - d3 >= 0. && d5 - d6 >= 0.) {
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    for (d = 0; d < 3; ++d) q[d] = b[d] + w * (c[d] - b[d]);
  } else {
    denom = 1. / (va + vb + vc);
    v     = vb * denom;
    w     = vc * denom;
    for (d = 0; d < 3; ++d) q[d] = a[d] + ab[d] * v + ac[d] * w;
  }
  for (d = 0; d < 3; ++d) r[d] = p[d] - q[d];
  return PetscSqrtReal(Dot3_Private(r, r));
}

/* Edge function of the directed edge (p, q) in the (y, z) plane, relative to the ray at the origin */
static PetscReal EdgeFunction_Private(const PetscReal p[], const PetscReal q[])
{
  return p[0] * q[1] - p[1] * q[0];
}

/* Sign of the edge function with exact zeros broken by moving the ray point to d = (delta, delta^2), delta -> 0+.
   E is bilinear, so E(p - d, q - d) = E(p, q) - E(d, q - p) = E(p, q) - delta (q.z - p.z) + delta^2 (q.y - p.y).
   For E(p, q) == 0 the sign is that of p.z - q.z, else of q.y - p.y. Both E and this rule are exactly antisymmetric
   in (p, q), so triangles sharing an edge or a vertex see opposite signs and the ray crosses exactly one of them. */
static PetscInt EdgeSign_Private(PetscReal e, const PetscReal p[], const PetscReal q[])
{
  if (e > 0.) return 1;
  if (e < 0.) return -1;
  if (p[1] != q[1]) return p[1] > q[1] ? 1 : -1;
  if (p[0] != q[0]) return q[0] > p[0] ? 1 : -1;
  return 0;
}

static void PrimitiveBox_Private(const IBBody_Surface *surf, PetscInt p, PetscReal lo[], PetscReal hi[])
{
  PetscInt v, d;

  for (d = 0; d < surf->dim; ++d) {
    lo[d] = PETSC_MAX_REAL;
    hi[d] = PETSC_MIN_REAL;
  }
  for (v = 0; v < surf->dim; ++v)
    for (d = 0; d < surf->dim; ++d) {
      lo[d] = PetscMin(lo[d], surf->v[surf->dim * surf->prim[surf->dim * p + v] + d]);
      hi[d] = PetscMax(hi[d], surf->v[surf->dim * surf->prim[surf->dim * p + v] + d]);
    }
}

/* Sign by parity along x-lines: phi = -1 inside, +1 outside */
static PetscErrorCode SampleSign_Private(const IBBody_Surface *surf, const PetscInt n[], const PetscReal *const coord[], const PetscReal lo[], const PetscReal hi[], PetscReal phi[])
{
  PetscInt        *cand, ncand = 0, p, i, j, k, c, v, ncross;
  PetscReal       *xc, plo[3], bhi[3], pt[3][2], e[3], y, z, sum;
  const PetscReal *a, *b, *x[3];

  PetscFunctionBegin;
  /* Primitives whose (y, z) extent overlaps the local (y, z) range can cross a local x-line */
  PetscCall(PetscMalloc1(surf->nprim, &cand));
  PetscCall(PetscMalloc1(surf->nprim, &xc));
  for (p = 0; p < surf->nprim; ++p) {
    PrimitiveBox_Private(surf, p, plo, bhi);
    if (bhi[1] < lo[1] || plo[1] > hi[1]) continue;
    if (surf->dim == 3 && (bhi[2] < lo[2] || plo[2] > hi[2])) continue;
    cand[ncand++] = p;
  }
  for (k = 0; k < n[2]; ++k)
    for (j = 0; j < n[1]; ++j) {
      y      = coord[1][j];
      z      = surf->dim == 3 ? coord[2][k] : 0.;
      ncross = 0;
      for (c = 0; c < ncand; ++c) {
        p = cand[c];
        if (surf->dim == 2) {
          a = &surf->v[2 * surf->prim[2 * p]];
          b = &surf->v[2 * surf->prim[2 * p + 1]];
          /* Half-open rule counts a vertex on the line exactly once */
          if ((a[1] > y) != (b[1] > y)) xc[ncross++] = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]);
        } else {
          for (v = 0; v < 3; ++v) {
            x[v]     = &surf->v[3 * surf->prim[3 * p + v]];
            pt[v][0] = x[v][1] - y;
            pt[v][1] = x[v][2] - z;
          }
          /* e[v] is the edge function of the edge opposite to vertex v */
          for (v = 0; v < 3; ++v) e[v] = EdgeFunction_Private(pt[(v + 1) % 3], pt[(v + 2) % 3]);
          sum = e[0] + e[1] + e[2];
          if (sum == 0.) continue; /* zero projected area */
          if (EdgeSign_Private(e[0], pt[1], pt[2]) != EdgeSign_Private(e[1], pt[2], pt[0]) || EdgeSign_Private(e[1], pt[2], pt[0]) != EdgeSign_Private(e[2], pt[0], pt[1])) continue;
          xc[ncross++] = (e[0] * x[0][0] + e[1] * x[1][0] + e[2] * x[2][0]) / sum;
        }
      }
      PetscCall(PetscSortReal(ncross, xc));
      c = 0;
      for (i = 0; i < n[0]; ++i) {
        while (c < ncross && xc[c] < coord[0][i]) ++c;
        phi[(k * n[1] + j) * n[0] + i] = (c % 2) ? -1. : 1.;
      }
    }
  PetscCall(PetscFree(xc));
  PetscCall(PetscFree(cand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Scale phi by the distance to the surface, clamped to the band; buckets are at least one band wide */
static PetscErrorCode SampleDistance_Private(const IBBody_Surface *surf, const PetscInt n[], const PetscReal *const coord[], const PetscReal lo[], const PetscReal hi[], PetscReal band, PetscReal phi[])
{
  PetscInt  dim = surf->dim, nb[3] = {1, 1, 1}, nbtot, p, d, i, j, k, b[3], bl[3], bh[3], q[3], bi, c;
  PetscInt *start, *list = NULL, *fill = NULL;
  PetscReal cs[3] = {1., 1., 1.}, plo[3], bhi[3], x[3] = {0., 0., 0.}, dist;

  PetscFunctionBegin;
  for (d = 0; d < dim; ++d) {
    nb[d] = PetscMax(1, (PetscInt)PetscFloorReal((hi[d] - lo[d]) / band));
    cs[d] = hi[d] > lo[d] ? (hi[d] - lo[d]) / nb[d] : band;
  }
  nbtot = nb[0] * nb[1] * nb[2];
  PetscCall(PetscCalloc1(nbtot + 1, &start));
  /* Two passes: count, then fill a CSR list of primitives per bucket */
  for (c = 0; c < 2; ++c) {
    if (c == 1) {
      for (bi = 0; bi < nbtot; ++bi) start[bi + 1] += start[bi];
      PetscCall(PetscMalloc1(start[nbtot], &list));
      PetscCall(PetscCalloc1(nbtot, &fill));
    }
    for (p = 0; p < surf->nprim; ++p) {
      PrimitiveBox_Private(surf, p, plo, bhi);
      for (d = 0; d < dim; ++d) {
        if (bhi[d] < lo[d] - band || plo[d] > hi[d] + band) break;
        bl[d] = PetscMin(PetscMax((PetscInt)PetscFloorReal((plo[d] - lo[d]) / cs[d]), 0), nb[d] - 1);
        bh[d] = PetscMin(PetscMax((PetscInt)PetscFloorReal((bhi[d] - lo[d]) / cs[d]), 0), nb[d] - 1);
      }
      if (d < dim) continue;
      for (d = dim; d < 3; ++d) bl[d] = bh[d] = 0;
      for (q[2] = bl[2]; q[2] <= bh[2]; ++q[2])
        for (q[1] = bl[1]; q[1] <= bh[1]; ++q[1])
          for (q[0] = bl[0]; q[0] <= bh[0]; ++q[0]) {
            bi = (q[2] * nb[1] + q[1]) * nb[0] + q[0];
            if (c == 0) ++start[bi + 1];
            else list[start[bi] + fill[bi]++] = p;
          }
    }
  }
  for (k = 0; k < n[2]; ++k)
    for (j = 0; j < n[1]; ++j)
      for (i = 0; i < n[0]; ++i) {
        x[0] = coord[0][i];
        if (dim > 1) x[1] = coord[1][j];
        if (dim > 2) x[2] = coord[2][k];
        for (d = 0; d < 3; ++d) {
          b[d]  = d < dim ? PetscMin(PetscMax((PetscInt)PetscFloorReal((x[d] - lo[d]) / cs[d]), 0), nb[d] - 1) : 0;
          bl[d] = PetscMax(b[d] - 1, 0);
          bh[d] = PetscMin(b[d] + 1, nb[d] - 1);
        }
        dist = band;
        for (q[2] = bl[2]; q[2] <= bh[2]; ++q[2])
          for (q[1] = bl[1]; q[1] <= bh[1]; ++q[1])
            for (q[0] = bl[0]; q[0] <= bh[0]; ++q[0]) {
              bi = (q[2] * nb[1] + q[1]) * nb[0] + q[0];
              for (c = start[bi]; c < start[bi + 1]; ++c) {
                p = list[c];
                if (dim == 2) dist = PetscMin(dist, DistancePointSegment_Private(x, &surf->v[2 * surf->prim[2 * p]], &surf->v[2 * surf->prim[2 * p + 1]]));
                else dist = PetscMin(dist, DistancePointTriangle_Private(x, &surf->v[3 * surf->prim[3 * p]], &surf->v[3 * surf->prim[3 * p + 1]], &surf->v[3 * surf->prim[3 * p + 2]]));
              }
            }
        phi[(k * n[1] + j) * n[0] + i] *= dist;
      }
  PetscCall(PetscFree(fill));
  PetscCall(PetscFree(list));
  PetscCall(PetscFree(start));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IBBodySample_Surface(IBBody body, PetscInt dim, const PetscInt n[], const PetscReal *const coord[], PetscReal phi[])
{
  IBBody_Surface *surf  = (IBBody_Surface *)body->data;
  PetscReal       lo[3] = {0., 0., 0.}, hi[3] = {0., 0., 0.}, h = 0., band;
  PetscInt        d, i;

  PetscFunctionBegin;
  PetscCheck(dim == surf->dim, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_INCOMP, "Surface is %" PetscInt_FMT "D but the mesh is %" PetscInt_FMT "D", surf->dim, dim);
  for (d = 0; d < dim; ++d) {
    lo[d] = coord[d][0];
    hi[d] = coord[d][n[d] - 1];
    for (i = 0; i + 1 < n[d]; ++i) h = PetscMax(h, coord[d][i + 1] - coord[d][i]);
  }
  band = h > 0. ? IBBODY_SURFACE_BAND * h : 1.;
  PetscCall(SampleSign_Private(surf, n, coord, lo, hi, phi));
  PetscCall(SampleDistance_Private(surf, n, coord, lo, hi, band, phi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IBBodySetUp_Surface(IBBody body)
{
  IBBody_Surface *surf = (IBBody_Surface *)body->data;

  PetscFunctionBegin;
  PetscCheck(surf->dim && surf->nprim > 0, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_WRONGSTATE, "Surface not set. Call IBBodySurfaceSetPolyline(), IBBodySurfaceSetTriangles() or IBBodySurfaceLoadSTL() first");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IBBodyReleaseGeometry_Surface(IBBody body)
{
  IBBody_Surface *surf = (IBBody_Surface *)body->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(surf->v));
  PetscCall(PetscFree(surf->prim));
  surf->nv    = 0;
  surf->nprim = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IBBodyView_Surface(IBBody body, PetscViewer viewer)
{
  IBBody_Surface *surf = (IBBody_Surface *)body->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerASCIIPrintf(viewer, "Surface: %" PetscInt_FMT "D, %" PetscInt_FMT " vertices, %" PetscInt_FMT " %s\n", surf->dim, surf->nv, surf->nprim, surf->dim == 3 ? "triangles" : "segments"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IBBodyDestroy_Surface(IBBody body)
{
  PetscFunctionBegin;
  PetscCall(IBBodyReleaseGeometry_Surface(body));
  PetscCall(PetscFree(body->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodyCreate_Surface(IBBody body)
{
  IBBody_Surface *surf;

  PetscFunctionBegin;
  PetscCall(PetscNew(&surf));
  body->data                 = (void *)surf;
  body->ops->setup           = IBBodySetUp_Surface;
  body->ops->sample          = IBBodySample_Surface;
  body->ops->releasegeometry = IBBodyReleaseGeometry_Surface;
  body->ops->destroy         = IBBodyDestroy_Surface;
  body->ops->view            = IBBodyView_Surface;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetSurface_Private(IBBody body, PetscInt dim, PetscInt nv, const PetscReal x[], PetscInt nprim, const PetscInt prim[])
{
  IBBody_Surface *surf;
  PetscInt        n;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(body, IBBODY_CLASSID, 1, IBBODYSURFACE);
  PetscCheck(!body->sampled, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_WRONGSTATE, "Cannot change the geometry of a body that a Mesh has sampled");
  for (n = 0; n < dim * nprim; ++n) PetscCheck(prim[n] >= 0 && prim[n] < nv, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_OUTOFRANGE, "Vertex index %" PetscInt_FMT " out of range [0, %" PetscInt_FMT ")", prim[n], nv);
  PetscCall(IBBodyReleaseGeometry_Surface(body));
  surf        = (IBBody_Surface *)body->data;
  surf->dim   = dim;
  surf->nv    = nv;
  surf->nprim = nprim;
  PetscCall(PetscMalloc1(dim * nv, &surf->v));
  PetscCall(PetscMalloc1(dim * nprim, &surf->prim));
  PetscCall(PetscArraycpy(surf->v, x, dim * nv));
  PetscCall(PetscArraycpy(surf->prim, prim, dim * nprim));
  body->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Every rank holds the full surface until the Mesh releases it after sampling in MeshSetUp() */
PetscErrorCode IBBodySurfaceSetPolyline(IBBody body, PetscInt n, const PetscReal xy[])
{
  PetscInt *seg, s;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(body, IBBODY_CLASSID, 1, IBBODYSURFACE);
  PetscCheck(n >= 3, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_OUTOFRANGE, "A closed polyline needs at least 3 vertices, got %" PetscInt_FMT, n);
  PetscAssertPointer(xy, 3);
  PetscCall(PetscMalloc1(2 * n, &seg));
  for (s = 0; s < n; ++s) {
    seg[2 * s]     = s;
    seg[2 * s + 1] = (s + 1) % n;
  }
  PetscCall(SetSurface_Private(body, 2, n, xy, n, seg));
  PetscCall(PetscFree(seg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodySurfaceSetTriangles(IBBody body, PetscInt nv, const PetscReal x[], PetscInt nt, const PetscInt tri[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecificType(body, IBBODY_CLASSID, 1, IBBODYSURFACE);
  PetscCheck(nv >= 3 && nt >= 1, PetscObjectComm((PetscObject)body), PETSC_ERR_ARG_OUTOFRANGE, "A triangle surface needs at least 3 vertices and 1 triangle");
  PetscAssertPointer(x, 3);
  PetscAssertPointer(tri, 5);
  PetscCall(SetSurface_Private(body, 3, nv, x, nt, tri));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IBBodySurfaceLoadSTL(IBBody body, const char filename[])
{
  MPI_Comm    comm;
  PetscMPIInt rank, cnt;
  PetscInt    nt = 0, n, *tri;
  PetscReal  *x  = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecificType(body, IBBODY_CLASSID, 1, IBBODYSURFACE);
  PetscAssertPointer(filename, 2);
  PetscCall(PetscObjectGetComm((PetscObject)body, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  /* Rank 0 reports a file error through nt, so every rank raises it after the broadcast */
  if (rank == 0) PetscCall(IBBodySurfaceReadSTL_Internal(filename, &nt, &x));
  PetscCallMPI(MPI_Bcast(&nt, 1, MPIU_INT, 0, comm));
  PetscCheck(nt != IBBODY_STL_ERR_OPEN, comm, PETSC_ERR_FILE_OPEN, "Cannot open STL file %s", filename);
  PetscCheck(nt != IBBODY_STL_ERR_ENDIAN, comm, PETSC_ERR_SUP, "Binary STL is only supported on little-endian hosts: %s", filename);
  PetscCheck(nt != IBBODY_STL_ERR_TRUNCATED, comm, PETSC_ERR_FILE_READ, "Truncated binary STL file %s", filename);
  PetscCheck(nt >= 0, comm, PETSC_ERR_FILE_UNEXPECTED, "Malformed ASCII STL file %s: a bad vertex line, or a vertex count that is not a positive multiple of 3", filename);
  if (rank != 0) PetscCall(PetscMalloc1(9 * nt, &x));
  PetscCall(PetscMPIIntCast(9 * nt, &cnt));
  PetscCallMPI(MPI_Bcast(x, cnt, MPIU_REAL, 0, comm));
  PetscCall(PetscMalloc1(3 * nt, &tri));
  for (n = 0; n < 3 * nt; ++n) tri[n] = n;
  PetscCall(IBBodySurfaceSetTriangles(body, 3 * nt, x, nt, tri));
  PetscCall(PetscFree(tri));
  PetscCall(PetscFree(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}
