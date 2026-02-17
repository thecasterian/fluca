---
name: add-tutorial
description: Scaffold a new tutorial example under fluca/tutorials/. Generates the source file, registers it in CMakeLists.txt, and verifies it builds and runs.
---

# Add Tutorial

Scaffold a new tutorial example following Fluca conventions.

## Required Input

Ask the user for:
1. **Module** — tutorial subdirectory (e.g., `fd`)
2. **Description** — what the tutorial demonstrates
3. **Dependencies** — which `fluca::*` libraries to link

## Steps

### 1. Determine next file number

```bash
ls fluca/tutorials/<module>/ex*.c
```

### 2. Generate source

```c
#include <flucasys.h>
/* Module headers as needed */

static const char help[] = "<Description>\n"
                           "Options:\n"
                           "  -<opt> <type>  : <desc>\n";

int main(int argc, char **argv)
{
  /* Declarations at block top */

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* 1. Parse options
     2. Create DM(s)
     3. Create and configure objects
     4. Solve / compute
     5. Output results (ViewFromOptions or PetscPrintf)
     6. Destroy in reverse creation order */

  PetscCall(FlucaFinalize());
}
```

Tutorials differ from tests:
- No `/*TEST*/` block — tutorials are not registered with ctest
- Typically longer, with comments explaining each step
- Use `SetFromOptions` / `ViewFromOptions` for runtime configurability
- May use PETSc solvers (SNES, KSP) as part of the example

### 3. Register in CMakeLists.txt

Add to `TUTORIAL_SRCS` in `fluca/tutorials/<module>/CMakeLists.txt`:

```cmake
set(TUTORIAL_SRCS
    ex1.c
    ...
    ex<N>.c   # <-- add here
)
```

For a new module, also:
- Create `fluca/tutorials/<module>/CMakeLists.txt` following the pattern
- Add `add_subdirectory(<module>)` to `fluca/tutorials/CMakeLists.txt`

### 4. Build and verify

```bash
cmake --build build
./build/fluca/tutorials/<module>/ex<N> <args>
```

## Checklist

- [ ] Help string documents all custom options
- [ ] Key steps have explanatory comments
- [ ] Source added to `TUTORIAL_SRCS` in CMakeLists.txt
- [ ] Builds and runs without errors
