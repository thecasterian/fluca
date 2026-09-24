#pragma once

#include <flucaphys.h>
#include <flucasys.h>
#include <petscsnes.h>

/* Seg - Segregated solver for the coupled system (13) of the theory guide.

   A Seg owns the time loop: it advances a Phys solution vector from the current time to the
   requested final time with a fixed time step, solving one coupled linear system per step.

   Seg is deliberately not a PETSc TS. The state it carries mixes time levels, since the velocity
   is at t^{n+1} while the pressure is at t^{n+1/2}, and the step is a linear solve that never
   evaluates an IFunction or an RHSFunction, so none of TS's contract applies.

   The final step is shortened, or slightly lengthened, so that the last time reached is exactly
   the maximum time, in the same way TS_EXACTFINALTIME_MATCHSTEP does it. */
typedef struct _p_Seg *Seg;

/* Seg types */
typedef const char *SegType;
#define SEGCNLINEAR "cnlinear" /* Linearized Crank-Nicolson */

typedef enum {
  SEG_CONVERGED_ITERATING   = 0,
  SEG_CONVERGED_TIME        = 1,
  SEG_CONVERGED_ITS         = 2,
  SEG_DIVERGED_LINEAR_SOLVE = -1,
} SegConvergedReason;
FLUCA_EXTERN const char *const *SegConvergedReasons;

FLUCA_EXTERN PetscClassId   SEG_CLASSID;
FLUCA_EXTERN PetscErrorCode SegInitializePackage(void);
FLUCA_EXTERN PetscErrorCode SegFinalizePackage(void);

/* Lifecycle */
FLUCA_EXTERN PetscErrorCode SegCreate(MPI_Comm, Seg *);
FLUCA_EXTERN PetscErrorCode SegSetType(Seg, SegType);
FLUCA_EXTERN PetscErrorCode SegGetType(Seg, SegType *);
FLUCA_EXTERN PetscErrorCode SegSetPhys(Seg, Phys);
FLUCA_EXTERN PetscErrorCode SegGetPhys(Seg, Phys *);
FLUCA_EXTERN PetscErrorCode SegSetFromOptions(Seg);
FLUCA_EXTERN PetscErrorCode SegSetUp(Seg);
FLUCA_EXTERN PetscErrorCode SegDestroy(Seg *);
FLUCA_EXTERN PetscErrorCode SegView(Seg, PetscViewer);
FLUCA_EXTERN PetscErrorCode SegViewFromOptions(Seg, PetscObject, const char[]);

/* Time loop parameters */
FLUCA_EXTERN PetscErrorCode SegSetTimeStep(Seg, PetscReal);
FLUCA_EXTERN PetscErrorCode SegGetTimeStep(Seg, PetscReal *);
FLUCA_EXTERN PetscErrorCode SegSetMaxTime(Seg, PetscReal);
FLUCA_EXTERN PetscErrorCode SegGetMaxTime(Seg, PetscReal *);
FLUCA_EXTERN PetscErrorCode SegSetMaxSteps(Seg, PetscInt);
FLUCA_EXTERN PetscErrorCode SegGetMaxSteps(Seg, PetscInt *);

/* Solving.

   SegSolve() projects the face velocity of the initial state once, before the first step, so that
   the state it starts from is discretely divergence free. Every SegSolve() call projects again,
   which makes repeated solves over the same interval reproduce each other exactly even when the
   caller refills the solution vector in between.

   SegStep() advances one step from the current state and does not project. */
FLUCA_EXTERN PetscErrorCode SegSolve(Seg, Vec);
FLUCA_EXTERN PetscErrorCode SegStep(Seg);
FLUCA_EXTERN PetscErrorCode SegGetSolution(Seg, Vec *);

/* Time loop state */
FLUCA_EXTERN PetscErrorCode SegSetTime(Seg, PetscReal);
FLUCA_EXTERN PetscErrorCode SegGetTime(Seg, PetscReal *);
FLUCA_EXTERN PetscErrorCode SegSetStepNumber(Seg, PetscInt);
FLUCA_EXTERN PetscErrorCode SegGetStepNumber(Seg, PetscInt *);
FLUCA_EXTERN PetscErrorCode SegSetConvergedReason(Seg, SegConvergedReason);
FLUCA_EXTERN PetscErrorCode SegGetConvergedReason(Seg, SegConvergedReason *);
FLUCA_EXTERN PetscErrorCode SegSetErrorIfStepFailed(Seg, PetscBool);
FLUCA_EXTERN PetscErrorCode SegGetErrorIfStepFailed(Seg, PetscBool *);

/* The solve of eq. (13) at each step, posed as a Picard iteration A x = b. Its options live under the
   -seg_snes_ prefix and those of its linear solve under -seg_ksp_ and -seg_pc_ */
FLUCA_EXTERN PetscErrorCode SegGetSNES(Seg, SNES *);

/* Called at the top of every step of SegSolve(), before the state is advanced */
FLUCA_EXTERN PetscErrorCode SegSetPreStep(Seg, PetscErrorCode (*)(Seg, void *), void *);

/* Options prefix */
FLUCA_EXTERN PetscErrorCode SegSetOptionsPrefix(Seg, const char[]);
FLUCA_EXTERN PetscErrorCode SegAppendOptionsPrefix(Seg, const char[]);
FLUCA_EXTERN PetscErrorCode SegGetOptionsPrefix(Seg, const char *[]);

/* Monitors */
FLUCA_EXTERN PetscErrorCode SegMonitorSet(Seg, PetscErrorCode (*)(Seg, void *), void *, PetscErrorCode (*)(void **));
FLUCA_EXTERN PetscErrorCode SegMonitorCancel(Seg);
FLUCA_EXTERN PetscErrorCode SegMonitor(Seg);
FLUCA_EXTERN PetscErrorCode SegMonitorSetFromOptions(Seg, const char[], const char[], const char[], PetscErrorCode (*)(Seg, PetscViewerAndFormat *), PetscErrorCode (*)(Seg, PetscViewerAndFormat *));
FLUCA_EXTERN PetscErrorCode SegMonitorDefault(Seg, PetscViewerAndFormat *);

/* Registration */
FLUCA_EXTERN PetscFunctionList SegList;
FLUCA_EXTERN PetscErrorCode    SegRegister(const char[], PetscErrorCode (*)(Seg));
FLUCA_EXTERN PetscErrorCode    SegRegisterAll(void);

/* Approximate block factorization preconditioner for the coupled system (13) */
#define PCABF "abf"

FLUCA_EXTERN PetscErrorCode PCABFSetFields(PC, PetscInt, PetscInt, PetscInt);
FLUCA_EXTERN PetscErrorCode PCABFGetSubKSPs(PC, KSP *, KSP *);

typedef enum {
  PC_ABF_AINV_ID,
  PC_ABF_AINV_DIAG,
  PC_ABF_AINV_ROWSUM,
} PCABFAinvType;
FLUCA_EXTERN const char *const PCABFAinvTypes[];

FLUCA_EXTERN PetscErrorCode PCABFSetSchurComplementAinvType(PC, PCABFAinvType);
FLUCA_EXTERN PetscErrorCode PCABFSetUpperTriangularAinvType(PC, PCABFAinvType);
