#include <flucasys.h>

const char *help = "hello, world program\n";

int main(int argc, char **argv) {
    PetscCall(FlucaInitialize(&argc, &argv, NULL, help));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Hello, world!\n"));
    PetscCall(FlucaFinalize());
}
