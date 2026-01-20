#include <fluca/private/flucafdimpl.h>

PetscErrorCode FlucaFDStencilLocationToDMStagStencilLocation_Internal(FlucaFDStencilLocation loc, DMStagStencilLocation *stag_loc)
{
  PetscFunctionBegin;
  switch (loc) {
  case FLUCAFD_ELEMENT:
    *stag_loc = DMSTAG_ELEMENT;
    break;
  case FLUCAFD_LEFT:
    *stag_loc = DMSTAG_LEFT;
    break;
  case FLUCAFD_DOWN:
    *stag_loc = DMSTAG_DOWN;
    break;
  case FLUCAFD_BACK:
    *stag_loc = DMSTAG_BACK;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported FlucaFDStencilLocation");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDRemoveZeroStencilPoints_Internal(PetscInt *ncols, DMStagStencil col[], PetscScalar v[])
{
  PetscScalar       v_abssum;
  PetscInt          ncols_new, c;
  PetscBool         remove;
  const PetscScalar atol = 1e-10, rtol = 1e-8;

  PetscFunctionBegin;
  v_abssum = 0.;
  for (c = 0; c < *ncols; ++c) v_abssum += PetscAbs(v[c]);
  ncols_new = 0;
  for (c = 0; c < *ncols; ++c) {
    remove = PetscAbs(v[c]) < atol || PetscAbs(v[c] / v_abssum) < rtol;
    if (!remove) {
      col[ncols_new] = col[c];
      v[ncols_new]   = v[c];
      ++ncols_new;
    }
  }
  *ncols = ncols_new;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkCreate_Internal(FlucaFDTermLink *term)
{
  FlucaFDTermLink t;
  PetscInt        d;

  PetscFunctionBegin;
  PetscCall(PetscNew(&t));
  for (d = 0; d < FLUCAFD_MAX_DIM; ++d) {
    t->deriv_order[d] = -1;
    t->accu_order[d]  = PETSC_INT_MAX;
  }
  t->input_loc = FLUCAFD_ELEMENT;
  t->input_c   = 0;
  t->next      = NULL;
  *term        = t;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkDuplicate_Internal(FlucaFDTermLink src, FlucaFDTermLink *dst)
{
  PetscInt d;

  PetscFunctionBegin;
  PetscCall(FlucaFDTermLinkCreate_Internal(dst));
  for (d = 0; d < FLUCAFD_MAX_DIM; ++d) {
    (*dst)->deriv_order[d] = src->deriv_order[d];
    (*dst)->accu_order[d]  = src->accu_order[d];
  }
  (*dst)->input_loc = src->input_loc;
  (*dst)->input_c   = src->input_c;
  (*dst)->next      = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkAppend_Internal(FlucaFDTermLink *link, FlucaFDTermLink term)
{
  FlucaFDTermLink curr;

  PetscFunctionBegin;
  if (!*link) {
    *link = term;
  } else {
    curr = *link;
    while (curr->next) curr = curr->next;
    curr->next = term;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaFDTermInfoEqual_Private(FlucaFDTermLink a, FlucaFDTermLink b, PetscBool *equal)
{
  PetscInt d;

  PetscFunctionBegin;
  *equal = PETSC_TRUE;
  for (d = 0; d < FLUCAFD_MAX_DIM; ++d)
    if (a->deriv_order[d] != b->deriv_order[d] || a->accu_order[d] != b->accu_order[d]) *equal = PETSC_FALSE;
  if (a->input_loc != b->input_loc || a->input_c != b->input_c) *equal = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkFind_Internal(FlucaFDTermLink link, FlucaFDTermLink term, PetscBool *found)
{
  FlucaFDTermLink t;
  PetscBool       equal;

  PetscFunctionBegin;
  *found = PETSC_FALSE;
  for (t = link; t; t = t->next) {
    PetscCall(FlucaFDTermInfoEqual_Private(t, term, &equal));
    if (equal) {
      *found = PETSC_TRUE;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaFDTermLinkDestroy_Internal(FlucaFDTermLink *link)
{
  FlucaFDTermLink curr, next;

  PetscFunctionBegin;
  curr = *link;
  while (curr) {
    next = curr->next;
    PetscCall(PetscFree(curr));
    curr = next;
  }
  *link = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
