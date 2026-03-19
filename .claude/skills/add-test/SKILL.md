---
name: add-test
description: Add a new test file or new test cases to an existing file in Fluca. Handles source generation, CMakeLists.txt registration, golden output capture, and ctest verification.
---

# Add Test

Add tests following Fluca's golden-output testing conventions.

## Required Input

Ask the user for:
1. **Module** — test subdirectory (e.g., `fd`, `cavity_flow`)
2. **Mode** — new file (`ex<N>.c`) or new cases in an existing file
3. **Description** — what the test exercises
4. **Test cases** — suffix names and command-line args for each case

## Mode A: New Test File

### 1. Determine next file number

```bash
ls fluca/tests/<module>/ex*.c
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

  /* 1. Create DM(s)
     2. Create and configure objects under test
     3. SetFromOptions
     4. SetUp
     5. Exercise and print results
     6. Destroy in reverse creation order */

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: <name>
    nsize: 1
    args: <args>

TEST*/
```

### 3. Register in CMakeLists.txt

Add the file to `TEST_SRCS` in `fluca/tests/<module>/CMakeLists.txt`.

For a new module, also create its `CMakeLists.txt` and add `add_subdirectory(<module>)` to `fluca/tests/CMakeLists.txt`.

### 4. Build, capture, verify

```bash
cmake --build build
./build/fluca/tests/<module>/ex<N> <args> > fluca/tests/<module>/output/ex<N>_<suffix>.out
cmake build && ctest --test-dir build -R "ex<N>" --output-on-failure
```

## Mode B: Add Cases to Existing File

1. Read the existing file
2. Append `test:` entries to the `/*TEST*/` block (before `TEST*/`)
3. Build, capture golden output, re-configure, verify

## Suffix Naming

- Use `snake_case`, descriptive of the specific scenario being tested
- Keep concise but unambiguous — a suffix should identify the case without reading the args
- Study existing suffixes in the same `ex*.c` file and follow their convention

Examples from the codebase: `first_deriv`, `second_deriv_left_bc_dirichlet`, `all_second_deriv_all_loc_down`, `vanleer`, `cavity_flow_2d`

## Checklist

- [ ] Test validates an internal algorithm directly — no SNES or TS solve required
- [ ] Inputs are constructed analytically; expected outputs are known without running a solver
- [ ] `/*TEST*/` block has `suffix` and `args`
- [ ] Source added to `TEST_SRCS` in CMakeLists.txt
- [ ] Golden output captured from actual run (never hand-written)
- [ ] All new tests pass via `ctest`
