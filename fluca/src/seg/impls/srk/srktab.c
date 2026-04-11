#include "srktab.h"
#include <flucaseg.h>
#include <petsc/private/petscimpl.h>

static SRKTableauLink SRKTableauList        = NULL;
static PetscBool      SRKRegisterAllCalled  = PETSC_FALSE;
static PetscBool      SRKPackageInitialized = PETSC_FALSE;

/* --- Registration -------------------------------------------------------- */

PetscErrorCode SegSRKRegister(const char name[], PetscInt order, PetscInt s, const PetscReal A[], const PetscReal At[], const PetscReal b[], const PetscReal bt[], const PetscReal c[], const PetscReal ct[], PetscReal alpha_tau_max)
{
  SRKTableauLink link;
  SRKTableau     t;
  PetscInt       i, j;
  PetscBool      match;

  PetscFunctionBegin;
  /* Reject duplicates */
  for (link = SRKTableauList; link; link = link->next) {
    PetscCall(PetscStrcmp(link->tab.name, name, &match));
    PetscCheck(!match, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "SRK tableau \"%s\" already registered", name);
  }

  PetscCall(PetscNew(&link));
  t = &link->tab;
  PetscCall(PetscStrallocpy(name, &t->name));
  t->order = order;
  t->s     = s;

  PetscCall(PetscMalloc3(s * s, &t->A, s * s, &t->At, s, &t->b));
  PetscCall(PetscMalloc3(s, &t->bt, s, &t->c, s, &t->ct));
  PetscCall(PetscArraycpy(t->A, A, s * s));
  PetscCall(PetscArraycpy(t->At, At, s * s));

  /* b: use supplied values, or copy last row of A */
  if (b) {
    PetscCall(PetscArraycpy(t->b, b, s));
  } else {
    PetscCall(PetscArraycpy(t->b, &A[(s - 1) * s], s));
  }
  /* bt: use supplied values, or copy last row of At */
  if (bt) {
    PetscCall(PetscArraycpy(t->bt, bt, s));
  } else {
    PetscCall(PetscArraycpy(t->bt, &At[(s - 1) * s], s));
  }

  /* c: use supplied values, or compute row sums of A */
  if (c) {
    PetscCall(PetscArraycpy(t->c, c, s));
  } else {
    for (i = 0; i < s; i++) {
      t->c[i] = 0.;
      for (j = 0; j < s; j++) t->c[i] += A[i * s + j];
    }
  }
  /* ct: use supplied values, or compute row sums of At */
  if (ct) {
    PetscCall(PetscArraycpy(t->ct, ct, s));
  } else {
    for (i = 0; i < s; i++) {
      t->ct[i] = 0.;
      for (j = 0; j < s; j++) t->ct[i] += At[i * s + j];
    }
  }

  /* Compute metadata flags */
  t->stiffly_accurate = PETSC_TRUE;
  for (i = 0; i < s; i++) {
    if (t->A[(s - 1) * s + i] != t->b[i]) {
      t->stiffly_accurate = PETSC_FALSE;
      break;
    }
  }

  t->explicit_stiffly_accurate = PETSC_TRUE;
  for (i = 0; i < s; i++) {
    if (t->At[(s - 1) * s + i] != t->bt[i]) {
      t->explicit_stiffly_accurate = PETSC_FALSE;
      break;
    }
  }

  t->explicit_first_stage = PETSC_TRUE;
  for (i = 0; i < s; i++) {
    if (t->A[i] != 0.) {
      t->explicit_first_stage = PETSC_FALSE;
      break;
    }
  }

  t->fsal = t->explicit_first_stage && t->stiffly_accurate;

  t->alpha_tau_max = alpha_tau_max;

  /* ARS type: first column of A is entirely zero (excluding A[0][0] which is always 0) */
  t->ars_type = PETSC_TRUE;
  for (i = 1; i < s; i++) {
    if (A[i * s] != 0.) {
      t->ars_type = PETSC_FALSE;
      break;
    }
  }

  /* Prepend to list */
  link->next     = SRKTableauList;
  SRKTableauList = link;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKLookupTableau_Internal(const char name[], SRKTableau *tab)
{
  SRKTableauLink link;
  PetscBool      match;

  PetscFunctionBegin;
  PetscAssertPointer(name, 1);
  PetscAssertPointer(tab, 2);
  for (link = SRKTableauList; link; link = link->next) {
    PetscCall(PetscStrcmp(link->tab.name, name, &match));
    if (match) {
      *tab = &link->tab;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown SRK tableau type \"%s\"", name);
}

/* --- Tableau definitions ------------------------------------------------- */

PetscErrorCode SegSRKRegisterAll(void)
{
  PetscFunctionBegin;
  if (SRKRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  SRKRegisterAllCalled = PETSC_TRUE;

  /* IMEX RK classification (Ascher, Ruuth, Spiteri 1997):
     - Type CK: A has the block form [[0,0],[a,A_hat]] where A_hat is an invertible matrix of size s-1,
       i.e., the first stage is explicit (A[0][0] = 0).
     - Type ARS: subtype of CK where a = 0, i.e., the first column of A is entirely zero.
     All schemes registered here are type CK or ARS with an explicit first stage. */

  /* ARS(1,1,1): 1st order, s=2
     Ascher, Ruuth, Spiteri (1997) Section 2.1
     Forward-backward Euler */
  {
    const PetscReal A[2][2] = {
      {0., 0.},
      {0., 1.}
    };
    const PetscReal At[2][2] = {
      {0., 0.},
      {1., 0.}
    };

    PetscCall(SegSRKRegister(SEGSRKARS111, 1, 2, &A[0][0], &At[0][0], NULL, NULL, NULL, NULL, 1.));
  }

  /* ARS(1,2,1): 1st order, s=2
     Ascher, Ruuth, Spiteri (1997) Section 2.2
     Forward-backward Euler; same A/At as ARS(1,1,1) but explicit weights bt differ from last row of At */
  {
    const PetscReal A[2][2] = {
      {0., 0.},
      {0., 1.}
    };
    const PetscReal At[2][2] = {
      {0., 0.},
      {1., 0.}
    };
    const PetscReal bt[2] = {0., 1.};

    PetscCall(SegSRKRegister(SEGSRKARS121, 1, 2, &A[0][0], &At[0][0], NULL, bt, NULL, NULL, 2.));
  }

  /* ARS(2,2,2): 2nd order, s=3
     Ascher, Ruuth, Spiteri (1997) Section 2.6
     L-stable SDIRK */
  {
    const PetscReal gamma   = (2. - PetscSqrtReal(2.)) / 2.;
    const PetscReal delta   = 1. - 1. / (2. * gamma);
    const PetscReal A[3][3] = {
      {0., 0.,         0.   },
      {0., gamma,      0.   },
      {0., 1. - gamma, gamma}
    };
    const PetscReal At[3][3] = {
      {0.,    0.,         0.},
      {gamma, 0.,         0.},
      {delta, 1. - delta, 0.}
    };

    PetscCall(SegSRKRegister(SEGSRKARS222, 2, 3, &A[0][0], &At[0][0], NULL, NULL, NULL, NULL, 0.82));
  }

  /* ARS(2,3,2): 2nd order, s=3
     Ascher, Ruuth, Spiteri (1997) Section 2.5
     L-stable SDIRK; explicit weights bt differ from last row of At */
  {
    const PetscReal gamma   = (2. - PetscSqrtReal(2.)) / 2.;
    const PetscReal delta   = -2. * PetscSqrtReal(2.) / 3.;
    const PetscReal A[3][3] = {
      {0., 0.,         0.   },
      {0., gamma,      0.   },
      {0., 1. - gamma, gamma}
    };
    const PetscReal At[3][3] = {
      {0.,    0.,         0.},
      {gamma, 0.,         0.},
      {delta, 1. - delta, 0.}
    };
    const PetscReal bt[3] = {0, 1. - gamma, gamma};

    PetscCall(SegSRKRegister(SEGSRKARS232, 2, 3, &A[0][0], &At[0][0], NULL, bt, NULL, NULL, 2.));
  }

  /* ARS(3,4,3): 3rd order, s=4, L-stable SDIRK
     Ascher, Ruuth, Spiteri (1997) Section 2.7
     gamma = middle root of 6x^3 - 18x^2 + 9x - 1; explicit weights bt differ from last row of At */
  {
    const PetscReal gamma   = 0.43586652150845967;
    const PetscReal b1      = -1.5 * gamma * gamma + 4. * gamma - .25;
    const PetscReal b2      = 1.5 * gamma * gamma - 5. * gamma + 1.25;
    const PetscReal A[4][4] = {
      {0., 0.,                0.,    0.   },
      {0., gamma,             0.,    0.   },
      {0., (1. - gamma) / 2., gamma, 0.   },
      {0., b1,                b2,    gamma}
    };
    const PetscReal At[4][4] = {
      {0.,               0.,              0.,              0.},
      {gamma,            0.,              0.,              0.},
      {.32127888627204,  .39665437448219, 0.,              0.},
      {-.10585829580000, .55292914790000, .55292914790000, 0.}
    };
    const PetscReal bt[4] = {0., b1, b2, gamma};

    PetscCall(SegSRKRegister(SEGSRKARS343, 3, 4, &A[0][0], &At[0][0], NULL, bt, NULL, NULL, 2.));
  }

  /* ARS(4,4,3): 3rd order, s=5, L-stable SDIRK, gamma = 1/2
     Ascher, Ruuth, Spiteri (1997) Section 2.8. PETSc TSARKIMEXARS443 */
  {
    const PetscReal A[5][5] = {
      {0., 0.,       0.,       0.,      0.     },
      {0., 1. / 2.,  0.,       0.,      0.     },
      {0., 1. / 6.,  1. / 2.,  0.,      0.     },
      {0., -1. / 2., 1. / 2.,  1. / 2., 0.     },
      {0., 3. / 2.,  -3. / 2., 1. / 2., 1. / 2.}
    };
    const PetscReal At[5][5] = {
      {0.,        0.,       0.,      0.,       0.},
      {1. / 2.,   0.,       0.,      0.,       0.},
      {11. / 18., 1. / 18., 0.,      0.,       0.},
      {5. / 6.,   -5. / 6., 1. / 2., 0.,       0.},
      {1. / 4.,   7. / 4.,  3. / 4., -7. / 4., 0.}
    };

    PetscCall(SegSRKRegister(SEGSRKARS443, 3, 5, &A[0][0], &At[0][0], NULL, NULL, NULL, NULL, 2.));
  }

  /* MARS(3,4,3): 3rd order, s=4
     Boscarino & Russo (2007)
     explicit weights bt differ from last row of At */
  {
    const PetscReal gamma   = .435866521508458;
    const PetscReal b2      = 1.20849664917601;
    const PetscReal b3      = -.64436317068446;
    const PetscReal A[4][4] = {
      {0., 0.,              0.,    0.   },
      {0., gamma,           0.,    0.   },
      {0., .28206673924577, gamma, 0.   },
      {0., b2,              b3,    gamma}
    };
    const PetscReal At[4][4] = {
      {0.,               0.,               0.,               0.},
      {.87173304301691,  0.,               0.,               0.},
      {.535396540307354, .182536720446875, 0.,               0.},
      {.63041255815287,  -.83193390106308, 1.20152134291021, 0.}
    };
    const PetscReal bt[4] = {0., b2, b3, gamma};

    PetscCall(SegSRKRegister(SEGSRKMARS343, 3, 4, &A[0][0], &At[0][0], NULL, bt, NULL, NULL, 2.));
  }

  /* MARK3(2)4L[2]SA: 3rd order, s=4
     Boscarino & Russo (2007)
     explicit weights bt differ from last row of At */
  {
    const PetscReal gamma   = .43586652150845;
    const PetscReal b1      = .60424832458800;
    const PetscReal b3      = .04011484609646;
    const PetscReal A[4][4] = {
      {0.,                0.,               0.,    0.   },
      {.43586652150845,   gamma,            0.,    0.   },
      {-4.30002662176923, 2.26541338346372, gamma, 0.   },
      {b1,                0.,               b3,    gamma}
    };
    const PetscReal At[4][4] = {
      {0.,                0.,               0.,              0.},
      {.871733043016917,  0.,               0.,              0.},
      {-3.06478674186224, 1.46604002506519, 0.,              0.},
      {.21444560762133,   .71075364965269,  .07480074272597, 0.}
    };
    const PetscReal bt[4] = {b1, 0., b3, gamma};

    PetscCall(SegSRKRegister(SEGSRKMARK324L2SA, 3, 4, &A[0][0], &At[0][0], NULL, bt, NULL, NULL, 2.));
  }

  /* ARK3(2)4L[2]SA: 3rd order, s=4
     Kennedy & Carpenter (2003)
     Same as PETSc TSARKIMEX3 */
  {
    const PetscReal A[4][4] = {
      {0.,                               0.,                               0.,                                0.                             },
      {1767732205903. / 4055673282236.,  1767732205903. / 4055673282236.,  0.,                                0.                             },
      {2746238789719. / 10658868560708., -640167445237. / 6845629431997.,  1767732205903. / 4055673282236.,   0.                             },
      {1471266399579. / 7840856788654.,  -4482444167858. / 7529755066697., 11266239266428. / 11593286722821., 1767732205903. / 4055673282236.}
    };
    const PetscReal At[4][4] = {
      {0.,                               0.,                               0.,                                0.},
      {1767732205903. / 2027836641118.,  0.,                               0.,                                0.},
      {5535828885825. / 10492691773637., 788022342437. / 10882634858940.,  0.,                                0.},
      {6485989280629. / 16251701735622., -4246266847089. / 9704473918619., 10755448449292. / 10357097424841., 0.}
    };

    PetscCall(SegSRKRegister(SEGSRKARK324L2SA, 3, 4, &A[0][0], &At[0][0], NULL, NULL, NULL, NULL, 2.));
  }

  /* ARK4(3)6L[2]SA: 4th order, s=6
     Kennedy & Carpenter (2003)
     Same as PETSc TSARKIMEX4 */
  {
    const PetscReal A[6][6] = {
      {0.,                           0.,                      0.,                      0.,                  0.,             0.     },
      {1. / 4.,                      1. / 4.,                 0.,                      0.,                  0.,             0.     },
      {8611. / 62500.,               -1743. / 31250.,         1. / 4.,                 0.,                  0.,             0.     },
      {5012029. / 34652500.,         -654441. / 2922500.,     174375. / 388108.,       1. / 4.,             0.,             0.     },
      {15267082809. / 155376265600., -71443401. / 120774400., 730878875. / 902184768., 2285395. / 8070912., 1. / 4.,        0.     },
      {82889. / 524892.,             0.,                      15625. / 83664.,         69875. / 102672.,    -2260. / 8211., 1. / 4.}
    };
    const PetscReal At[6][6] = {
      {0.,                              0.,                                0.,                                0.,                               0.,             0.},
      {1. / 2.,                         0.,                                0.,                                0.,                               0.,             0.},
      {13861. / 62500.,                 6889. / 62500.,                    0.,                                0.,                               0.,             0.},
      {-116923316275. / 2393684061468., -2731218467317. / 15368042101831., 9408046702089. / 11113171139209.,  0.,                               0.,             0.},
      {-451086348788. / 2902428689909., -2682348792572. / 7519795681897.,  12662868775082. / 11960479115383., 3355817975965. / 11060851509271., 0.,             0.},
      {647845179188. / 3216320057751.,  73281519250. / 8382639484533.,     552539513391. / 3454668386233.,    3354512671639. / 8306763924573.,  4040. / 17871., 0.}
    };

    PetscCall(SegSRKRegister(SEGSRKARK436L2SA, 4, 6, &A[0][0], &At[0][0], NULL, NULL, NULL, NULL, 2.));
  }

  /* ARK5(4)8L[2]SA: 5th order, s=8
     Kennedy & Carpenter (2003)
     Same as PETSc TSARKIMEX5 */
  {
    const PetscReal A[8][8] = {
      {0.,                               0.,                               0.,                                0.,                                 0.,                               0.,                                 0.,                                0.        },
      {41. / 200.,                       41. / 200.,                       0.,                                0.,                                 0.,                               0.,                                 0.,                                0.        },
      {41. / 400.,                       -567603406766. / 11931857230679., 41. / 200.,                        0.,                                 0.,                               0.,                                 0.,                                0.        },
      {683785636431. / 9252920307686.,   0.,                               -110385047103. / 1367015193373.,   41. / 200.,                         0.,                               0.,                                 0.,                                0.        },
      {3016520224154. / 10081342136671., 0.,                               30586259806659. / 12414158314087., -22760509404356. / 11113319521817., 41. / 200.,                       0.,                                 0.,                                0.        },
      {218866479029. / 1489978393911.,   0.,                               638256894668. / 5436446318841.,    -1179710474555. / 5321154724896.,   -60928119172. / 8023461067671.,   41. / 200.,                         0.,                                0.        },
      {1020004230633. / 5715676835656.,  0.,                               25762820946817. / 25263940353407., -2161375909145. / 9755907335909.,   -211217309593. / 5846859502534.,  -4269925059573. / 7827059040749.,   41. / 200.,                        0.        },
      {-872700587467. / 9133579230613.,  0.,                               0.,                                22348218063261. / 9555858737531.,   -1143369518992. / 8141816002931., -39379526789629. / 19018526304540., 32727382324388. / 42900044865799., 41. / 200.}
    };
    const PetscReal At[8][8] = {
      {0.,                                 0.,                             0.,                                0.,                                 0.,                              0.,                                0.,                              0.},
      {41. / 100.,                         0.,                             0.,                                0.,                                 0.,                              0.,                                0.,                              0.},
      {367902744464. / 2072280473677.,     677623207551. / 8224143866563., 0.,                                0.,                                 0.,                              0.,                                0.,                              0.},
      {1268023523408. / 10340822734521.,   0.,                             1029933939417. / 13636558850479.,  0.,                                 0.,                              0.,                                0.,                              0.},
      {14463281900351. / 6315353703477.,   0.,                             66114435211212. / 5879490589093.,  -54053170152839. / 4284798021562.,  0.,                              0.,                                0.,                              0.},
      {14090043504691. / 34967701212078.,  0.,                             15191511035443. / 11219624916014., -18461159152457. / 12425892160975., -281667163811. / 9011619295870., 0.,                                0.,                              0.},
      {19230459214898. / 13134317526959.,  0.,                             21275331358303. / 2942455364971.,  -38145345988419. / 4862620318723.,  -1. / 8.,                        -1. / 8.,                          0.,                              0.},
      {-19977161125411. / 11928030595625., 0.,                             -40795976796054. / 6384907823539., 177454434618887. / 12078138498510., 782672205425. / 8267701900261.,  -69563011059811. / 9646580694205., 7356628210526. / 4942186776405., 0.}
    };

    PetscCall(SegSRKRegister(SEGSRKARK548L2SA, 5, 8, &A[0][0], &At[0][0], NULL, NULL, NULL, NULL, 2.));
  }

  /* BHR(5,5,3): 3rd order, s=5
     Boscarino, Pareschi, Russo (2009) Appendix method 1
     gamma ≈ same as ARS(3,4,3); this simplifies tableaus much */
  {
    const PetscReal gamma   = 0.43586652150845967;
    const PetscReal c2      = 2. * gamma;
    const PetscReal c3      = 2. * gamma;
    const PetscReal c4      = 2.3402125751086804; /* Free parameter */
    const PetscReal b3      = (1. / 3. - c4 / 2. - gamma * (1. - c4)) / (c3 * (c3 - c4));
    const PetscReal b4      = (1. / 3. - c3 / 2. - gamma * (1. - c3)) / (c4 * (c4 - c3));
    const PetscReal b1      = 1. - b3 - b4 - gamma;
    const PetscReal a41     = 1.5 * c4 - (c4 * c4) / (4. * gamma) - gamma;
    const PetscReal a43     = (c4 * c4) / (4. * gamma) - c4 / 2.;
    const PetscReal at41    = c4 - (c4 * c4) / (4. * gamma);
    const PetscReal at43    = (c4 * c4) / (4. * gamma);
    const PetscReal at54    = (1. / (12. * gamma) - c4 * c4 * b4 - gamma - 4. * gamma * gamma * b3) / (c4 * (c4 - 2. * gamma));
    const PetscReal at53    = 1. / (4. * gamma) + b3 - c4 / (2. * gamma) * at54;
    const PetscReal at51    = 1. + b3 - at53 - at54;
    const PetscReal A[5][5] = {
      {0.,    0.,    0.,    0.,    0.   },
      {gamma, gamma, 0.,    0.,    0.   },
      {gamma, 0.,    gamma, 0.,    0.   },
      {a41,   0.,    a43,   gamma, 0.   },
      {b1,    0.,    b3,    b4,    gamma}
    };
    const PetscReal At[5][5] = {
      {0.,         0.,    0.,   0.,   0.},
      {2. * gamma, 0.,    0.,   0.,   0.},
      {gamma,      gamma, 0.,   0.,   0.},
      {at41,       0.,    at43, 0.,   0.},
      {at51,       -b3,   at53, at54, 0.}
    };
    const PetscReal bt[5] = {b1, 0., b3, b4, gamma};

    PetscCall(SegSRKRegister(SEGSRKBHR553, 3, 5, &A[0][0], &At[0][0], NULL, bt, NULL, NULL, 2.));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKGetTableauList_Internal(SRKTableauLink *list)
{
  PetscFunctionBegin;
  *list = SRKTableauList;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKRegisterDestroy(void)
{
  SRKTableauLink link;
  SRKTableau     t;

  PetscFunctionBegin;
  while ((link = SRKTableauList)) {
    t              = &link->tab;
    SRKTableauList = link->next;
    PetscCall(PetscFree3(t->A, t->At, t->b));
    PetscCall(PetscFree3(t->bt, t->c, t->ct));
    PetscCall(PetscFree(t->name));
    PetscCall(PetscFree(link));
  }
  SRKRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --- Package lifecycle --------------------------------------------------- */

PetscErrorCode SegSRKInitializePackage(void)
{
  PetscFunctionBegin;
  if (SRKPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  SRKPackageInitialized = PETSC_TRUE;

  PetscCall(SegSRKRegisterAll());
  PetscCall(PetscRegisterFinalize(SegSRKFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SegSRKFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(SegSRKRegisterDestroy());
  SRKPackageInitialized = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
