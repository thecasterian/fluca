---
name: petsc-class
description: Scaffold a new PETSc-style class (full lifecycle, ops table, package registration, CMake) or add a new subtype (impl data struct, factory function, ops wiring, registration) in Fluca.
---

# New PETSc-Style Class

Scaffold a new PETSc-style class following PETSc's object-oriented framework. This generates all the boilerplate files needed to add a new polymorphic type with the standard lifecycle pattern.

## Required Input

Ask the user for:
1. **Class name** (PascalCase, e.g. `IB` for Immersed Boundary)
2. **Description** (short, e.g. "Immersed boundary")
3. **Custom ops** (virtual methods beyond the standard set)
4. **Custom fields** (data members for `struct _p_<Class>`)
5. **Dependencies** (which `fluca::*` libraries this links against)

## Files to Generate

Given a class named `Foo` (lowercase `foo`):

```
fluca/include/flucafoo.h                       # Public header
fluca/include/fluca/private/fooimpl.h          # Private implementation header
fluca/src/foo/interface/foobasic.c             # Create, Destroy, SetType, SetUp, View
fluca/src/foo/interface/fooopts.c              # SetFromOptions + property getters/setters
fluca/src/foo/interface/fooreg.c               # Register, RegisterAll
fluca/src/foo/interface/foopkg.c               # InitializePackage, FinalizePackage
fluca/src/foo/CMakeLists.txt                   # Library target
```

## Template: Public Header (`flucafoo.h`)

```c
#pragma once

#include <flucasys.h>
#include <petscsys.h>

typedef struct _p_Foo *Foo;

typedef const char *FooType;

FLUCA_EXTERN PetscErrorCode FooCreate(MPI_Comm, Foo *);
FLUCA_EXTERN PetscErrorCode FooSetType(Foo, FooType);
FLUCA_EXTERN PetscErrorCode FooGetType(Foo, FooType *);
FLUCA_EXTERN PetscErrorCode FooSetFromOptions(Foo);
FLUCA_EXTERN PetscErrorCode FooSetUp(Foo);
FLUCA_EXTERN PetscErrorCode FooView(Foo, PetscViewer);
FLUCA_EXTERN PetscErrorCode FooDestroy(Foo *);

FLUCA_EXTERN PetscErrorCode FooInitializePackage(void);
FLUCA_EXTERN PetscErrorCode FooFinalizePackage(void);

FLUCA_EXTERN PetscFunctionList FooList;
FLUCA_EXTERN PetscErrorCode    FooRegister(const char[], PetscErrorCode (*)(Foo));

FLUCA_EXTERN PetscClassId FOO_CLASSID;
```

## Template: Private Header (`fooimpl.h`)

```c
#pragma once

#include <fluca/private/flucaimpl.h>
#include <flucafoo.h>

FLUCA_EXTERN PetscBool      FooRegisterAllCalled;
FLUCA_EXTERN PetscErrorCode FooRegisterAll(void);
FLUCA_EXTERN PetscLogEvent  FOO_SetUp;

typedef struct _FooOps *FooOps;

struct _FooOps {
  PetscErrorCode (*setfromoptions)(Foo, PetscOptionItems);
  PetscErrorCode (*setup)(Foo);
  PetscErrorCode (*destroy)(Foo);
  PetscErrorCode (*view)(Foo, PetscViewer);
};

struct _p_Foo {
  PETSCHEADER(struct _FooOps);

  /* Parameters */

  /* Data */
  void *data; /* implementation-specific data */

  /* State */
  PetscBool setupcalled;
};
```

## Template: `foobasic.c`

```c
#include <fluca/private/fooimpl.h>

PetscClassId      FOO_CLASSID = 0;
PetscLogEvent     FOO_SetUp   = 0;
PetscFunctionList FooList              = NULL;
PetscBool         FooRegisterAllCalled = PETSC_FALSE;

PetscErrorCode FooCreate(MPI_Comm comm, Foo *foo)
{
  Foo f;

  PetscFunctionBegin;
  PetscAssertPointer(foo, 2);

  PetscCall(FooInitializePackage());
  PetscCall(FlucaHeaderCreate(f, FOO_CLASSID, "Foo", "Foo description", "Foo", comm, FooDestroy, FooView));

  f->data        = NULL;
  f->setupcalled = PETSC_FALSE;

  *foo = f;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FooSetType(Foo foo, FooType type)
{
  FooType            old_type;
  PetscErrorCode (*impl_create)(Foo);
  PetscBool          match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(foo, FOO_CLASSID, 1);

  PetscCall(FooGetType(foo, &old_type));
  PetscCall(PetscObjectTypeCompare((PetscObject)foo, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscFunctionListFind(FooList, type, &impl_create));
  PetscCheck(impl_create, PetscObjectComm((PetscObject)foo), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown Foo type: %s", type);

  if (old_type) {
    PetscTryTypeMethod(foo, destroy);
    PetscCall(PetscMemzero(foo->ops, sizeof(struct _FooOps)));
  }

  PetscCall(PetscObjectChangeTypeName((PetscObject)foo, type));
  PetscCall((*impl_create)(foo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FooGetType(Foo foo, FooType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(foo, FOO_CLASSID, 1);
  PetscCall(FooRegisterAll());
  *type = ((PetscObject)foo)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FooSetUp(Foo foo)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(foo, FOO_CLASSID, 1);
  if (foo->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscLogEventBegin(FOO_SetUp, (PetscObject)foo, 0, 0, 0));

  PetscCheck(((PetscObject)foo)->type_name, PetscObjectComm((PetscObject)foo), PETSC_ERR_ARG_WRONGSTATE, "Foo type not set. Call FooSetType() first");

  PetscTryTypeMethod(foo, setup);

  PetscCall(PetscLogEventEnd(FOO_SetUp, (PetscObject)foo, 0, 0, 0));

  foo->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FooView(Foo foo, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(foo, FOO_CLASSID, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)foo), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(foo, 1, viewer, 2);

  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)foo, viewer));
  PetscTryTypeMethod(foo, view, viewer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FooDestroy(Foo *foo)
{
  PetscFunctionBegin;
  if (!*foo) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific((*foo), FOO_CLASSID, 1);

  if (--((PetscObject)(*foo))->refct > 0) {
    *foo = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscTryTypeMethod((*foo), destroy);
  PetscCall(PetscHeaderDestroy(foo));
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

## Template: `fooopts.c`

```c
#include <fluca/private/fooimpl.h>

PetscErrorCode FooSetFromOptions(Foo foo)
{
  char      type[256];
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(foo, FOO_CLASSID, 1);
  PetscCall(FooRegisterAll());

  PetscObjectOptionsBegin((PetscObject)foo);
  PetscCall(PetscOptionsFList("-foo_type", "Foo type", "FooSetType", FooList,
                              (char *)(((PetscObject)foo)->type_name ? ((PetscObject)foo)->type_name : ""),
                              type, sizeof(type), &flg));
  if (flg) PetscCall(FooSetType(foo, type));
  PetscTryTypeMethod(foo, setfromoptions, PetscOptionsObject);
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Add property getters/setters here using the pattern:
 *
 * PetscErrorCode FooSetBar(Foo foo, PetscReal bar)
 * {
 *   PetscFunctionBegin;
 *   PetscValidHeaderSpecific(foo, FOO_CLASSID, 1);
 *   PetscCheck(!foo->setupcalled, ...);  // if pre-setup only
 *   foo->bar = bar;
 *   PetscFunctionReturn(PETSC_SUCCESS);
 * }
 */
```

## Template: `fooreg.c`

```c
#include <fluca/private/fooimpl.h>

/* Declare factory functions for each subtype:
 * FLUCA_EXTERN PetscErrorCode FooCreate_Bar(Foo);
 */

PetscErrorCode FooRegister(const char type[], PetscErrorCode (*function)(Foo))
{
  PetscFunctionBegin;
  PetscCall(FooInitializePackage());
  PetscCall(PetscFunctionListAdd(&FooList, type, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FooRegisterAll(void)
{
  PetscFunctionBegin;
  if (FooRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  /* PetscCall(FooRegister(FOOBAR, FooCreate_Bar)); */
  FooRegisterAllCalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

## Template: `foopkg.c`

```c
#include <fluca/private/fooimpl.h>

static PetscBool FooPackageInitialized = PETSC_FALSE;

PetscErrorCode FooInitializePackage(void)
{
  char         log_list[256];
  PetscBool    opt, pkg;
  PetscClassId classids[1];

  PetscFunctionBegin;
  if (FooPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  FooPackageInitialized = PETSC_TRUE;

  /* Register class */
  PetscCall(PetscClassIdRegister("Foo", &FOO_CLASSID));
  /* Register constructors */
  PetscCall(FooRegisterAll());
  /* Register events */
  PetscCall(PetscLogEventRegister("FooSetUp", FOO_CLASSID, &FOO_SetUp));

  /* Process Info */
  classids[0] = FOO_CLASSID;
  PetscCall(PetscInfoProcessClass("foo", 1, classids));
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", log_list, sizeof(log_list), &opt));
  if (opt) {
    PetscCall(PetscStrInList("foo", log_list, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(FOO_CLASSID));
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(FooFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FooFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&FooList));
  FooPackageInitialized = PETSC_FALSE;
  FooRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

## Template: `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20)

add_library(fluca_foo SHARED
    interface/foobasic.c
    interface/fooopts.c
    interface/fooreg.c
    interface/foopkg.c
)

target_include_directories(fluca_foo PUBLIC
    ${CMAKE_SOURCE_DIR}/fluca/include
)
target_link_libraries(fluca_foo PUBLIC
    fluca::sys  # adjust dependencies
)
add_library(fluca::foo ALIAS fluca_foo)
```

## Implementing a Subtype

Once the base class is scaffolded, concrete subtypes implement the ops (virtual methods). Given a base class `Foo` and a subtype named `Bar` (lowercase `bar`):

### Subtype File Layout

Place implementation files under `impl/<subtypename>/` alongside the base class `interface/` directory:

```
fluca/src/foo/
├── interface/              # Base class (already generated above)
│   ├── foobasic.c
│   ├── fooopts.c
│   ├── fooreg.c
│   └── foopkg.c
└── impl/
    └── bar/                # Subtype implementation
        └── bar.c           # FooCreate_Bar() + all ops implementations
```

If the subtype is large, split it into multiple files by concern (e.g. `bar.c`, `bario.c`, `barvec.c`).

The subtype also needs a private header for its data struct:

```
fluca/include/fluca/private/foobarimpl.h
```

### Template: Subtype Data Struct (`foobarimpl.h`)

Define a plain struct named `Foo_Bar` (base class underscore subtype name). This holds all state specific to the subtype. It is allocated in the factory function and stored in the base object's `void *data` field.

```c
#pragma once

#include <fluca/private/fooimpl.h>

typedef struct {
  /* Subtype-specific fields */
  PetscInt  n;
  PetscReal alpha;
  Vec       work;
  PetscBool ready;
} Foo_Bar;
```

### Template: Subtype Implementation (`bar.c`)

```c
#include <fluca/private/foobarimpl.h>

static PetscErrorCode FooSetFromOptions_Bar(Foo foo, PetscOptionItems PetscOptionsObject)
{
  Foo_Bar *bar = (Foo_Bar *)foo->data;

  PetscFunctionBegin;
  /* PetscCall(PetscOptionsInt("-foo_bar_n", "...", "FooBarSetN", bar->n, &bar->n, NULL)); */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FooSetUp_Bar(Foo foo)
{
  Foo_Bar *bar = (Foo_Bar *)foo->data;

  PetscFunctionBegin;
  /* Allocate work vectors, precompute data, etc. */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FooView_Bar(Foo foo, PetscViewer viewer)
{
  Foo_Bar *bar = (Foo_Bar *)foo->data;

  PetscFunctionBegin;
  /* Display subtype-specific info */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FooDestroy_Bar(Foo foo)
{
  Foo_Bar *bar = (Foo_Bar *)foo->data;

  PetscFunctionBegin;
  /* Free subtype-specific resources */
  PetscCall(VecDestroy(&bar->work));
  PetscCall(PetscFree(foo->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FooCreate_Bar(Foo foo)
{
  Foo_Bar *bar;

  PetscFunctionBegin;
  PetscCall(PetscNew(&bar));
  foo->data = (void *)bar;

  /* Initialize fields to safe defaults */
  bar->n     = 0;
  bar->alpha = 1.0;
  bar->work  = NULL;
  bar->ready = PETSC_FALSE;

  /* Wire ops table */
  foo->ops->setfromoptions = FooSetFromOptions_Bar;
  foo->ops->setup          = FooSetUp_Bar;
  foo->ops->destroy        = FooDestroy_Bar;
  foo->ops->view           = FooView_Bar;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

### Subtype Registration

Three changes are needed to register the subtype with the base class:

**1. Add the type string constant to the public header (`flucafoo.h`)**:

```c
#define FOOBAR "bar"
```

**2. Declare the factory function and register it in `fooreg.c`**:

```c
FLUCA_EXTERN PetscErrorCode FooCreate_Bar(Foo);

PetscErrorCode FooRegisterAll(void)
{
  PetscFunctionBegin;
  if (FooRegisterAllCalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(FooRegister(FOOBAR, FooCreate_Bar));
  FooRegisterAllCalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
```

**3. Add the source files to `CMakeLists.txt`**:

```cmake
add_library(fluca_foo SHARED
    interface/foobasic.c
    interface/fooopts.c
    interface/fooreg.c
    interface/foopkg.c
    impl/bar/bar.c
)
```

## Integration Steps

After generating the files:

1. **Register in parent CMake**: Add `add_subdirectory(foo)` to `fluca/src/CMakeLists.txt`
2. **Wire dependencies**: If other modules need this class, add `fluca::foo` to their `target_link_libraries`
3. **Verify build**: Run `make` from `build/` to confirm compilation

## Naming Conventions

Follow PETSc style throughout:
- **Types**: `PascalCase` (`Foo`, `FooType`)
- **Functions**: `PascalCase` with module prefix (`FooSetUp`, `FooDestroy`)
- **Class IDs**: `UPPER_SNAKE` with module prefix (`FOO_CLASSID`)
- **Event handlers**: `PascalCase` with uppercase module prefix (`FOO_SetUp`)
- **Type strings**: lowercase (`"bar"` registered via `#define FOOBAR "bar"`)
- **Ops struct**: `_FooOps` with `PetscErrorCode` function pointers
- **Object struct**: `_p_Foo` with `PETSCHEADER` as first member
- **Impl data struct**: `Foo_Bar` (class_subtype) allocated in factory function
- **Private functions**: `FLUCA_INTERN`, named `FooMethod_SubType` (e.g. `FooSetUp_Bar`)
- **Validation**: `PetscValidHeaderSpecific(foo, FOO_CLASSID, argnum)` in every public function
