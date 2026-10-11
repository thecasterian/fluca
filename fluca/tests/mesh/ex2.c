#include <flucaibbody.h>
#include <flucasys.h>
#include <fluca/private/ibbodyimpl.h>

static const char help[] = "Test IBBody: level-set sampling and body velocity evaluation\n"
                           "Options:\n"
                           "  -velocity <constant|function|function_fd> : how the body velocity is given (default: constant)\n";

static PetscErrorCode Circle_Private(PetscInt dim, const PetscReal x[], void *ctx, PetscReal *phi)
{
  PetscFunctionBeginUser;
  *phi = PetscSqrtReal((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) - 0.25;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Rigid rotation about (0.5, 0.5) with angular velocity omega(t) = t */
static PetscErrorCode Rotation_Private(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar u[])
{
  PetscFunctionBeginUser;
  u[0] = -t * (x[1] - 0.5);
  u[1] = t * (x[0] - 0.5);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RotationDot_Private(PetscInt dim, PetscReal t, const PetscReal x[], void *ctx, PetscScalar u[])
{
  PetscFunctionBeginUser;
  u[0] = -(x[1] - 0.5);
  u[1] = x[0] - 0.5;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscReal Clean_Private(PetscReal v)
{
  return PetscAbsReal(v) < 1e-12 ? 0. : v;
}

int main(int argc, char **argv)
{
  IBBody            body;
  char              velocity[64] = "constant";
  const PetscScalar u_const[2]   = {1., 2.};
  const PetscReal   c[3]         = {0., 0.5, 1.};
  const PetscReal  *coord[3]     = {c, c, NULL};
  const PetscInt    n[3]         = {3, 3, 1};
  const PetscReal   x[2]         = {1., 0.5};
  PetscReal         phi[9];
  PetscScalar       u[2], udot[2];
  PetscBool         isfn, isfnfd;
  PetscInt          i, j;

  PetscFunctionBeginUser;
  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-velocity", velocity, sizeof(velocity), NULL));
  PetscCall(PetscStrcmp(velocity, "function", &isfn));
  PetscCall(PetscStrcmp(velocity, "function_fd", &isfnfd));

  PetscCall(IBBodyCreate(PETSC_COMM_WORLD, &body));
  PetscCall(IBBodySetType(body, IBBODYLEVELSET));
  PetscCall(IBBodyLevelSetSetFunction(body, Circle_Private, NULL));
  if (isfn) PetscCall(IBBodySetVelocityFunction(body, Rotation_Private, RotationDot_Private, NULL));
  else if (isfnfd) PetscCall(IBBodySetVelocityFunction(body, Rotation_Private, NULL, NULL));
  else PetscCall(IBBodySetVelocity(body, 2, u_const));
  PetscCall(IBBodySetFromOptions(body));
  PetscCall(IBBodySetUp(body));
  PetscCall(IBBodyView(body, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(IBBodySample_Internal(body, 2, n, coord, phi));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "phi on {0, 0.5, 1}^2 (rows from y = 1 down):\n"));
  for (j = 2; j >= 0; --j) {
    for (i = 0; i < 3; ++i) PetscCall(PetscPrintf(PETSC_COMM_WORLD, " %.6f", (double)Clean_Private(phi[j * 3 + i])));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n"));
  }

  PetscCall(IBBodyEvaluateVelocity_Internal(body, 2, 2., x, PETSC_FALSE, u));
  PetscCall(IBBodyEvaluateVelocity_Internal(body, 2, 2., x, PETSC_TRUE, udot));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "u(x = (1, 0.5), t = 2) = (%.6f, %.6f)\n", (double)Clean_Private(PetscRealPart(u[0])), (double)Clean_Private(PetscRealPart(u[1]))));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "du/dt(x = (1, 0.5), t = 2) = (%.6f, %.6f)\n", (double)Clean_Private(PetscRealPart(udot[0])), (double)Clean_Private(PetscRealPart(udot[1]))));

  PetscCall(IBBodyDestroy(&body));
  PetscCall(FlucaFinalize());
  return 0;
}

/*TEST

  test:
    suffix: velocity_constant
    nsize: 1
    args: -velocity constant

  test:
    suffix: velocity_function
    nsize: 1
    args: -velocity function

  test:
    suffix: velocity_function_fd
    nsize: 1
    args: -velocity function_fd

TEST*/
