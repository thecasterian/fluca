#include <petscsys.h>

PetscScalar FlucaFDSecondOrderTVDLimiterSuperbee_Internal(PetscScalar r)
{
  PetscScalar a, b;

  a = PetscMin(2. * r, 1.);
  b = PetscMin(r, 2.);
  return PetscMax(0., PetscMax(a, b));
}

PetscScalar FlucaFDSecondOrderTVDLimiterMinmod_Internal(PetscScalar r)
{
  PetscScalar a;

  a = PetscMin(r, 1.);
  return PetscMax(0., a);
}

PetscScalar FlucaFDSecondOrderTVDLimiterMC_Internal(PetscScalar r)
{
  PetscScalar a;

  a = PetscMin(PetscMin(2. * r, (1. + r) / 2.), 2.);
  return PetscMax(0., a);
}

PetscScalar FlucaFDSecondOrderTVDLimiterVanLeer_Internal(PetscScalar r)
{
  PetscScalar a;

  a = PetscAbs(r);
  return (r + a) / (1. + a);
}

PetscScalar FlucaFDSecondOrderTVDLimiterVanAlbada_Internal(PetscScalar r)
{
  return r <= 0. ? 0. : (r * r + r) / (r * r + 1.);
}

PetscScalar FlucaFDSecondOrderTVDLimiterBarthJesperson_Internal(PetscScalar r)
{
  PetscScalar a, b;

  if (r <= 0.) return 0.;
  a = 4. * r / (1. + r);
  b = 4. / (1. + r);
  return (1. + r) / 2. * PetscMin(1., PetscMin(a, b));
}

PetscScalar FlucaFDSecondOrderTVDLimiterVenkatakrishnan_Internal(PetscScalar r)
{
  PetscScalar a, b;

  a = 4. * r * (3. * r + 1) / (11. * r * r + 4. * r + 1.);
  b = 4. * (r + 3.) / (r * r + 4. * r + 11.);

  return r <= 0. ? 0. : (1. + r) / 2. * PetscMin(a, b);
}

PetscScalar FlucaFDSecondOrderTVDLimiterUpwind_Internal(PetscScalar r)
{
  return 0.;
}

PetscScalar FlucaFDSecondOrderTVDLimiterSOU_Internal(PetscScalar r)
{
  return r;
}

PetscScalar FlucaFDSecondOrderTVDLimiterQUICK_Internal(PetscScalar r)
{
  return (3. + r) / 4.;
}

PetscScalar FlucaFDSecondOrderTVDLimiterKoren_Internal(PetscScalar r)
{
  PetscScalar a;

  a = PetscMin(PetscMin(2. * r, (1. + 2. * r) / 3.), 2.);
  return PetscMax(0., a);
}
