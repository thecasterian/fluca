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
- Use `/*TEST*/` blocks with **run-only** semantics (exit code 0 = pass, no golden output files)
- Parsed by `fluca_parse_tutorial_file()` in `FlucaTestUtils.cmake`, which calls `RunTutorial.cmake`
- Typically longer, with comments explaining each step
- Use `SetFromOptions` / `ViewFromOptions` for runtime configurability
- May use PETSc solvers (SNES, KSP) as part of the example

### 3. Add `/*TEST*/` block

Append a `/*TEST ... TEST*/` block at the end of the source file to register tutorial cases with ctest:

```c
/*TEST

  test:
    suffix: default
    nsize: 1

  test:
    suffix: high_peclet
    nsize: 1
    args: -gamma 0.01 -flucafd_limiter superbee

TEST*/
```

Each `test:` entry supports `suffix` (required), `nsize` (ignored), and `args` (optional). No `output_file` field — tutorials only check exit code.

### 4. Register in CMakeLists.txt

Add to `TUTORIAL_SRCS` in `fluca/tutorials/<module>/CMakeLists.txt`:

```cmake
set(TUTORIAL_SRCS
    ex1.c
    ...
    ex<N>.c   # <-- add here
)
```

The `fluca_parse_tutorial_file()` call in the foreach loop automatically picks up the new file.

For a new module, also:
- Create `fluca/tutorials/<module>/CMakeLists.txt` following the pattern (include `FlucaTestUtils` and call `fluca_parse_tutorial_file`)
- Add `add_subdirectory(<module>)` to `fluca/tutorials/CMakeLists.txt`

### 5. Build and verify

```bash
cmake build                                    # Re-configure to pick up new TEST blocks
cmake --build build                            # Build
ctest --test-dir build -R "tutorials_<module>" # Run tutorial test cases
```

## Checklist

- [ ] Help string documents all custom options
- [ ] Key steps have explanatory comments
- [ ] `/*TEST*/` block with at least one test case
- [ ] Source added to `TUTORIAL_SRCS` in CMakeLists.txt
- [ ] Builds and all tutorial test cases pass (exit code 0)
