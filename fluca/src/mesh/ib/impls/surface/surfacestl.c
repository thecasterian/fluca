#include <fluca/private/ibbodyimpl.h>

/* Binary STL: 80-byte header, little-endian uint32 count, 50 bytes per triangle (normal, 3 vertices as float32, uint16).
   A file whose size is 84 + 50 * count is binary; anything else is parsed as ASCII ("vertex x y z" lines). */
PetscErrorCode IBBodySurfaceReadSTL_Internal(const char filename[], PetscInt *nt, PetscReal **x)
{
  FILE         *fp;
  long          size;
  unsigned char header[80], cnt[4], rec[50];
  unsigned long count  = 0;
  unsigned int  one    = 1;
  PetscBool     binary = PETSC_FALSE, isvertex;
  PetscInt      t, m, nvert = 0, maxvert = 0;
  float         f;
  char          tok[256];
  double        v[3];
  PetscReal    *grown;

  PetscFunctionBegin;
  PetscCall(PetscFOpen(PETSC_COMM_SELF, filename, "rb", &fp));
  PetscCheck(fseek(fp, 0, SEEK_END) == 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Cannot seek in %s", filename);
  size = ftell(fp);
  rewind(fp);
  if (size >= 84 && fread(header, 1, 80, fp) == 80 && fread(cnt, 1, 4, fp) == 4) {
    count  = (unsigned long)cnt[0] | ((unsigned long)cnt[1] << 8) | ((unsigned long)cnt[2] << 16) | ((unsigned long)cnt[3] << 24);
    binary = (PetscBool)((unsigned long)size == 84 + 50 * count);
  }
  *x = NULL;
  if (binary) {
    PetscCheck(*(unsigned char *)&one == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Binary STL is only supported on little-endian hosts");
    *nt = (PetscInt)count;
    PetscCall(PetscMalloc1(9 * (*nt), x));
    for (t = 0; t < *nt; ++t) {
      PetscCheck(fread(rec, 1, 50, fp) == 50, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Truncated binary STL %s", filename);
      for (m = 0; m < 9; ++m) {
        PetscCall(PetscMemcpy(&f, rec + 12 + 4 * m, sizeof(float)));
        (*x)[9 * t + m] = (PetscReal)f;
      }
    }
  } else {
    rewind(fp);
    while (fscanf(fp, "%255s", tok) == 1) {
      PetscCall(PetscStrcmp(tok, "vertex", &isvertex));
      if (!isvertex) continue;
      PetscCheck(fscanf(fp, "%lf %lf %lf", &v[0], &v[1], &v[2]) == 3, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Malformed vertex in ASCII STL %s", filename);
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
    PetscCheck(nvert > 0 && nvert % 3 == 0, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "ASCII STL %s has %" PetscInt_FMT " vertices, not a positive multiple of 3", filename, nvert);
    *nt = nvert / 3;
  }
  PetscCall(PetscFClose(PETSC_COMM_SELF, fp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
