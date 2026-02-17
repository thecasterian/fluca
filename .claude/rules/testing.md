# Testing Conventions for Fluca

This project uses a **golden-output testing system** — not googletest. Tests are C executables that print results to stdout, compared against expected output files.

## Test File Structure

```
fluca/tests/<module>/
├── CMakeLists.txt          # Build + test registration
├── ex1.c                   # Test source (may contain multiple test cases)
├── ex2.c
├── <module>test.h          # Shared test helpers (optional)
└── output/
    ├── ex1_<suffix>.out    # Golden output for each test case
    └── ex2_<suffix>.out
```

Tests are named `ex<N>.c` with sequential numbering. Each file is a standalone `main()`.

## Test Source Template

```c
#include <flucafd.h>
#include <flucasys.h>
#include <petscdmstag.h>

#include "fdtest.h"   /* module-specific helpers, if available */

static const char help[] = "Brief description of what this tests\n"
                           "Options:\n"
                           "  -i <int>  : Description of option\n";

int main(int argc, char **argv)
{
  /* Declarations at block top */

  PetscCall(FlucaInitialize(&argc, &argv, NULL, help));

  /* Test body:
     1. Create DM
     2. Create objects under test
     3. Configure via SetFromOptions (allows test args to control behavior)
     4. SetUp
     5. Exercise and print results
     6. Destroy in reverse order */

  PetscCall(FlucaFinalize());
}

/*TEST

  test:
    suffix: descriptive_name
    nsize: 1
    args: -option1 value1 -option2 value2

  test:
    suffix: another_case
    nsize: 1
    args: -different_options here

TEST*/
```

Key points:
- Use `FlucaInitialize` / `FlucaFinalize` (not `PetscInitialize`).
- Destroy objects in **reverse creation order**.

## `/*TEST*/` Block Format

Each `test:` entry requires:

| Field | Required | Description |
|-------|----------|-------------|
| `suffix` | Yes | Appended to exe name for the CTest name and output file |
| `nsize` | No | MPI size (currently always 1, parsed but ignored) |
| `args` | No | Command-line arguments passed to the executable |
| `output_file` | No | Custom golden output path (default: `output/<base>_<suffix>.out`) |

The `/*TEST*/` block is parsed by `FlucaTestUtils.cmake` at **configure time**. After adding or modifying test blocks, re-run `cmake` to register them.

## Golden Output Files

- Stored in `output/` subdirectory alongside test sources.
- Named `<base>_<suffix>.out` (e.g., `ex1_first_deriv.out`).
- Content is the **exact stdout** of a successful run.
- Comparison is byte-exact via `cmake -E compare_files`.
- An `empty.out` file may exist for tests that produce no output.

### Creating a golden output file

Run the test manually and capture stdout:

```bash
./build/fluca/tests/<module>/<exN> <args> > fluca/tests/<module>/output/<exN>_<suffix>.out
```

## CMakeLists.txt Pattern

```cmake
cmake_minimum_required(VERSION 3.20)

include(FlucaTestUtils)

set(TEST_SRCS
    ex1.c
    ex2.c
)

foreach(test_src ${TEST_SRCS})
    get_filename_component(test_base_name ${test_src} NAME_WE)
    set(test_name tests_<module>_${test_base_name})
    add_executable(${test_name} ${test_src})
    set_target_properties(${test_name} PROPERTIES OUTPUT_NAME ${test_base_name})
    target_link_libraries(${test_name} PRIVATE
        fluca::<module>
        m
    )
    fluca_parse_test_file(${CMAKE_CURRENT_SOURCE_DIR}/${test_src} ${test_name} ${test_base_name})
endforeach()
```

To add a new test module, also add `add_subdirectory(<module>)` to `fluca/tests/CMakeLists.txt`.

## Running Tests

```bash
# Build then run all tests
cmake --build build && ctest --test-dir build

# Run specific tests by name pattern
ctest --test-dir build -R "ex1_first_deriv"

# Verbose output on failure
ctest --test-dir build --output-on-failure
```

## Adding a New Test Case

1. **Add a `test:` entry** in the `/*TEST*/` block of an existing `ex*.c`, or create a new `ex<N>.c`.
2. **Build**: `cmake --build build`
3. **Run manually** to verify output is correct:
   ```bash
   ./build/fluca/tests/<module>/<exN> <args>
   ```
4. **Capture golden output**:
   ```bash
   ./build/fluca/tests/<module>/<exN> <args> > fluca/tests/<module>/output/<exN>_<suffix>.out
   ```
5. **Re-configure** if you added new `test:` blocks: `cmake build`
6. **Verify**: `ctest --test-dir build -R "<suffix>"`
