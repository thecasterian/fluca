#include <fluca/private/ibbodyimpl.h>

/* Binary STL: 80-byte header, little-endian uint32 count, 50 bytes per triangle (normal, 3 vertices as float32, uint16).
   A file whose size is 84 + 50 * count is binary; anything else is parsed as ASCII ("vertex x y z" lines).
   A file error is not raised: *nt is set to an IBBODY_STL_ERR_* code and *x to NULL, so that the caller can share
   the outcome with the other ranks before raising it collectively. */
PetscErrorCode IBBodySurfaceReadSTL_Internal(const char filename[], PetscInt *nt, PetscReal **x)
{
  FILE         *fp;
  long          size;
  unsigned char header[80], cnt[4], rec[50];
  unsigned long count  = 0;
  unsigned int  one    = 1;
  PetscBool     binary = PETSC_FALSE, readable, isvertex;
  PetscInt      t, m, nvert = 0, maxvert = 0, err = 0;
  float         f;
  char          tok[256];
  double        v[3];
  PetscReal    *grown;

  PetscFunctionBegin;
  *nt = 0;
  *x  = NULL;
  PetscCall(PetscTestFile(filename, 'r', &readable));
  if (!readable) {
    *nt = IBBODY_STL_ERR_OPEN;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscFOpen(PETSC_COMM_SELF, filename, "rb", &fp));
  if (fseek(fp, 0, SEEK_END) != 0) err = IBBODY_STL_ERR_OPEN;
  else {
    size = ftell(fp);
    rewind(fp);
    if (size >= 84 && fread(header, 1, 80, fp) == 80 && fread(cnt, 1, 4, fp) == 4) {
      count  = (unsigned long)cnt[0] | ((unsigned long)cnt[1] << 8) | ((unsigned long)cnt[2] << 16) | ((unsigned long)cnt[3] << 24);
      binary = (unsigned long)size == 84 + 50 * count;
    }
  }
  if (!err && binary) {
    if (*(unsigned char *)&one != 1) err = IBBODY_STL_ERR_ENDIAN;
    else {
      PetscCall(PetscMalloc1(9 * (PetscInt)count, x));
      for (t = 0; t < (PetscInt)count; ++t) {
        if (fread(rec, 1, 50, fp) != 50) {
          err = IBBODY_STL_ERR_TRUNCATED;
          break;
        }
        for (m = 0; m < 9; ++m) {
          PetscCall(PetscMemcpy(&f, rec + 12 + 4 * m, sizeof(float)));
          (*x)[9 * t + m] = (PetscReal)f;
        }
      }
      *nt = (PetscInt)count;
    }
  } else if (!err) {
    rewind(fp);
    while (fscanf(fp, "%255s", tok) == 1) {
      PetscCall(PetscStrcmp(tok, "vertex", &isvertex));
      if (!isvertex) continue;
      if (fscanf(fp, "%lf %lf %lf", &v[0], &v[1], &v[2]) != 3) {
        err = IBBODY_STL_ERR_MALFORMED;
        break;
      }
      if (nvert == maxvert) {
        maxvert = maxvert ? 2 * maxvert : 96;
        PetscCall(PetscMalloc1(3 * maxvert, &grown));
        if (nvert) PetscCall(PetscArraycpy(grown, *x, 3 * nvert));
        PetscCall(PetscFree(*x));
        *x = grown;
      }
      for (m = 0; m < 3; ++m) (*x)[3 * nvert + m] = (PetscReal)v[m];
      ++nvert;
    }
    if (!err && (nvert == 0 || nvert % 3 != 0)) err = IBBODY_STL_ERR_MALFORMED;
    *nt = nvert / 3;
  }
  PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  if (err) {
    PetscCall(PetscFree(*x));
    *nt = err;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
