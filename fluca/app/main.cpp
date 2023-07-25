#include <petsc.h>

int main(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscPrintf(PETSC_COMM_WORLD, "Hello, world!\n");
    PetscFinalize();
}
