# Build Conventions for Fluca

## Quick Reference

```bash
cmake --build build              # Build
ctest --test-dir build -R tests_ # Run unit tests (fast)
cmake build                      # Re-configure (after adding new test blocks)
```

**Note:** Bare `ctest --test-dir build` runs both unit tests and tutorial tests. Tutorial tests are very slow (minutes each). Prefer `ctest -R tests_` for routine verification. Only run `ctest -R tutorials_` when verifying tutorial-related changes.

## Library Targets

Each module is a shared library with an alias target:

| Target | Alias | Dependencies |
|--------|-------|-------------|
| `fluca_sys` | `fluca::sys` | PETSc, HDF5, CGNS |
| `fluca_fd` | `fluca::fd` | `fluca::sys` |
| `fluca_mesh` | `fluca::mesh` | `fluca::sys` |
| `fluca_ns` | `fluca::ns` | `fluca::mesh` |
| `fluca_viewer` | `fluca::viewer` | `fluca::sys` |

Always link against `fluca::*` alias targets, not the raw `fluca_*` names.

## Adding Source Files to an Existing Module

Add the `.c` file to the `add_library()` call in `fluca/src/<module>/CMakeLists.txt`:

```cmake
add_library(fluca_<module> SHARED
    ...
    impls/<subtype>/<newfile>.c   # <-- add here
)
```

## Adding a New Module

1. Create `fluca/src/<module>/CMakeLists.txt`:
   ```cmake
   cmake_minimum_required(VERSION 3.20)

   add_library(fluca_<module> SHARED
       interface/<module>basic.c
       ...
   )
   target_include_directories(fluca_<module> PUBLIC
       ${CMAKE_SOURCE_DIR}/fluca/include
   )
   target_link_libraries(fluca_<module> PUBLIC
       fluca::sys
   )
   add_library(fluca::<module> ALIAS fluca_<module>)
   ```
2. Add `add_subdirectory(<module>)` to `fluca/src/CMakeLists.txt`
3. Place public header at `fluca/include/fluca<module>.h`
4. Place private header at `fluca/include/fluca/private/<module>impl.h`

## Tests and Tutorials

Tests and tutorials have their own `CMakeLists.txt` under `fluca/tests/` and `fluca/tutorials/`. Both include `FlucaTestUtils` and use `/*TEST*/` blocks parsed at configure time:

- **Tests** use `fluca_parse_test_file()` → `RunTest.cmake` (golden output comparison)
- **Tutorials** use `fluca_parse_tutorial_file()` → `RunTutorial.cmake` (exit code 0 only, no golden output)

```cmake
include(FlucaTestUtils)

set(<TYPE>_SRCS ex1.c ex2.c)
foreach(src ${<TYPE>_SRCS})
    ...
    target_link_libraries(... PRIVATE fluca::<module> m)
    fluca_parse_test_file(...)      # for tests
    fluca_parse_tutorial_file(...)  # for tutorials
endforeach()
```
