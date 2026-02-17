---
name: golden-update
description: Recapture golden output files for tests whose behavior changed intentionally. Identifies failing tests, regenerates .out files, and verifies all tests pass.
---

# Golden Update

Update golden output files after an intentional behavior change.

## Required Input

Ask the user for:
1. **Scope** — which tests to update:
   - All failing tests
   - Specific test pattern (e.g., `ex1`, `ex1_first_deriv`)
   - Specific module (e.g., `fd`)

## Steps

### 1. Build

```bash
cmake --build build
```

### 2. Identify failing tests

```bash
ctest --test-dir build -R "<pattern>" --output-on-failure 2>&1
```

Parse the output to find which tests failed. Each CTest name maps to a test suffix and executable:
- CTest name `tests_<module>_ex<N>_<suffix>` → executable `ex<N>`, suffix `<suffix>`

### 3. Find args for each failing test

Read the `/*TEST*/` block in the source file to find the `args:` for each failing suffix. The source file is at `fluca/tests/<module>/ex<N>.c`.

### 4. Recapture golden output

For each failing test, run the executable and capture stdout:

```bash
./build/fluca/tests/<module>/ex<N> <args> > fluca/tests/<module>/output/ex<N>_<suffix>.out
```

**Before overwriting**, show the diff between old and new output to the user for confirmation:

```bash
./build/fluca/tests/<module>/ex<N> <args> > /tmp/ex<N>_<suffix>.out
diff fluca/tests/<module>/output/ex<N>_<suffix>.out /tmp/ex<N>_<suffix>.out
```

Only overwrite after the user confirms the new output is correct.

### 5. Verify

```bash
ctest --test-dir build -R "<pattern>" --output-on-failure
```

All updated tests must pass.

## Checklist

- [ ] Changes are intentional (not a regression)
- [ ] Diff reviewed and confirmed by user before overwriting
- [ ] All updated tests pass after recapture
- [ ] No unrelated tests broken
