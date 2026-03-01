# PETSc Coding Conventions for Fluca

This project follows PETSc style. All C code must conform to these rules.

## Language

- Pure **C** (not C++). Intersection of C99, C++11, MSVC v1900+.
- No variable-length arrays. No `assert.h` (use `PetscAssert()`). No `register`. No `rand()`.
- **Never** use `(void)param;` casts to suppress unused-parameter warnings. Leave unused parameters as-is; the compiler warnings are suppressed project-wide via build flags.

## Naming

| Entity | Convention | Example |
|--------|-----------|---------|
| Functions | `PascalCase` with module prefix | `FlucaFDSetUp()`, `MeshCartCreate2d()` |
| Types | `PascalCase` | `FlucaFD`, `FlucaFDType` |
| Enum values / macros | `UPPER_SNAKE_CASE` | `MAT_FINAL_ASSEMBLY`, `FLUCAFD_CLASSID` |
| Type string constants | `UPPER_SNAKE` prefix + lowercase name | `#define FLUCAFDDERIVATIVE "derivative"` |
| Ops struct members | Lowercase, no prefix | `setup`, `destroy`, `setfromoptions` |
| Options database keys | Lowercase with underscores | `-flucafd_deriv_order` |
| Subtype functions | `Base_Subtype` | `FlucaFDSetUp_Derivative()` |
| Private/internal | Append `_Internal`, `_Private`, or class/type name | `FlucaFDTermLinkDestroy_Internal()` |
| Function pointer typedefs | End in `Fn` | `SNESFunctionFn` |

Function prototypes in headers **exclude parameter names** — types only.

## Symbol Visibility

| Scope | Macro |
|-------|-------|
| Public API (in public headers) | `FLUCA_EXTERN` |
| Internal across translation units (in headers) | `FLUCA_INTERN` |
| File-local | `static` |

Visibility macros (`FLUCA_EXTERN`, `FLUCA_INTERN`) are only used in **header** declarations. Source file definitions do not need them — the declaration in the header is sufficient. Never use bare `extern`.

## Function Structure

Every function returning `PetscErrorCode` must follow this skeleton:

```c
PetscErrorCode FlucaFDDoSomething(FlucaFD fd, PetscInt n)
{
  PetscInt i;            /* all locals declared at block top */

  PetscFunctionBegin;    /* immediately after declarations + blank line */
  /* ... body ... */
  PetscFunctionReturn(PETSC_SUCCESS);  /* only way to return */
}
```

Rules:
- **Blank line** between last declaration and `PetscFunctionBegin;`.
- **No blank line** immediately after `PetscFunctionBegin;` or before `PetscFunctionReturn()`.
- **Never** use bare `return` in a function that has `PetscFunctionBegin`.
- **Never** place a function call inside `PetscFunctionReturn(...)`.
- Use `PetscFunctionBeginUser` in `main()` / example code.
- Void-returning functions: `PetscFunctionReturnVoid()`.

## Error Handling

```c
/* Correct — wrap every PETSc call */
PetscCall(VecCreate(comm, &v));

/* Wrong — never store and check separately */
PetscErrorCode ierr = VecCreate(comm, &v);
PetscCall(ierr);
```

- `PetscCall()` for all PETSc/Fluca function calls.
- `PetscCheck(condition, comm, error_code, "message", ...)` for runtime assertions.
- `PetscAssert()` for debug-only assertions (replaces `assert()`).
- Begin/end macro pairs (`PetscOptionsBegin`/`PetscOptionsEnd`) handle errors internally — do **not** wrap them.

## Argument Validation

All **public** functions must validate arguments between `PetscFunctionBegin` and the body:

```c
PetscFunctionBegin;
PetscValidHeaderSpecific(fd, FLUCAFD_CLASSID, 1);   /* PETSc object arg */
PetscAssertPointer(ptr, 2);                         /* non-null pointer arg */
PetscValidLogicalCollectiveInt(fd, n, 3);           /* collective int arg */
```

The number is the 1-based argument position.

## Variable Declarations

- All locals at **block top** (C90 style).
- Group variables of the same type on the same or adjacent lines.
- Pointers of different arity are different types (separate lines).
- Arrays of different size are different types.
- Don't declare pointers and non-pointers in a single line. Same for arrays and non-arrays.

## Memory Management

```c
/* Allocate */
PetscCall(PetscNew(&bar));           /* single struct */
PetscCall(PetscMalloc1(n, &arr));    /* array */
PetscCall(PetscCalloc1(n, &arr));    /* zero-initialized array */

/* Free */
PetscCall(PetscFree(arr));
```

**Never** use `malloc()`, `calloc()`, `free()` — only PETSc allocation functions.

## Formatting

Formatting is enforced by `.clang-format` (PETSc style) and pre-commit hooks.

Key points for writing new code:
- **Indentation**: 2 spaces (no tabs).
- **Single-statement `if`/`else`/`for`**: No braces.
  ```c
  if (fd->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  ```
- **Braces**: Required only for multi-statement blocks.
- **Preprocessor**: `#if defined(X)` over `#ifdef X`. Better: `PetscDefined(X)` at runtime.
- **Header guards**: `#pragma once` as first non-comment line.
- **Floating-point literals**: No trailing zeros after decimal point.
- **Enums**: Always add a trailing comma after the last enumerator.

## Comments

```c
/* Single-line block comment with spaces at both ends */

/* Multi-line comment starts with space.
   No asterisks at line beginnings.
   Ends with space before closing. */

// Single-line C++ style comment is only used for TODOs and clang-format directives
// TODO: something to do
// clang-format off
```

- **Never** `/*No space*/`.
- Proper grammar and spelling.

## Printf Format Specifiers

```c
PetscInt  n;
PetscReal x;

PetscCall(PetscPrintf(comm, "n = %" PetscInt_FMT "\n", n));    /* PetscInt */
PetscCall(PetscPrintf(comm, "x = %g\n", (double)x));           /* PetscReal: cast to double */
```

## Object Lifecycle

All PETSc-style objects in Fluca follow: **Create → SetType → SetFromOptions → SetUp → Use → Destroy**.

```c
FlucaFD fd;
PetscCall(FlucaFDCreate(comm, &fd));
PetscCall(FlucaFDSetType(fd, FLUCAFDDERIVATIVE));
PetscCall(FlucaFDSetFromOptions(fd));
PetscCall(FlucaFDSetUp(fd));
/* ... use fd ... */
PetscCall(FlucaFDDestroy(&fd));   /* Destroy takes pointer-to-pointer */
```

## Logging

- Call `PetscLogFlops()` directly with the flop count. Never accumulate in a local counter.
- Register events with `PetscLogEventRegister()` and bracket with `PetscLogEventBegin/End`.

## Callback Function Typedefs

When a function accepts a callback (function pointer) parameter, always define a separate `typedef` for the **function type** (not a pointer type). The typedef name must end in `Fn`. Unlike regular function prototypes, callback typedefs **include parameter names** for documentation.

```c
/* Define the function type (NOT a pointer — no (*)) */
typedef PetscErrorCode PhysComputeRHSFn(DM dm, Vec u, Vec F, void *ctx);

/* Use as a pointer in the API */
FLUCA_EXTERN PetscErrorCode PhysSetComputeRHS(Phys, PhysComputeRHSFn *, void *);

/* Use as a pointer in structs */
PhysComputeRHSFn *computerhs;
```

Never use raw function pointer syntax directly in a function signature.

## Quick Checklist

- [ ] `PascalCase` functions with module prefix, `UPPER_SNAKE_CASE` enums/macros, suffixes for internal/private symbols
- [ ] `PetscFunctionBegin` / `PetscFunctionReturn(PETSC_SUCCESS)` in every function
- [ ] All PETSc calls wrapped in `PetscCall()`
- [ ] Public functions validate args with `PetscValidHeaderSpecific` etc.
- [ ] `FLUCA_EXTERN` / `FLUCA_INTERN` in header declarations, `static` for file-local symbols in source files
- [ ] Memory via `PetscNew` / `PetscMalloc*` / `PetscFree` only
- [ ] `PetscInt` for indices/sizes, `PetscScalar` for floating point
- [ ] Format specifiers: `PetscInt_FMT`, cast `PetscReal` to `double`
- [ ] No bare `return`, no `malloc`/`free`, no `assert()`, no VLAs
- [ ] Locals at block top, blank line before `PetscFunctionBegin`, no blank line before `PetscFunctionReturn`
- [ ] No braces for single-statement `if`/`else`/`for`
- [ ] No trailing zeros after decimal point
- [ ] Callback parameters use a named `typedef` ending in `Fn` — no raw function pointer syntax in signatures
