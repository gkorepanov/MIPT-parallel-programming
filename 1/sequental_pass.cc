#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto x = 0;

    if (rank != 0) {
        MPI_Recv(&x, 1, MPI_INT,
                 rank - 1, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    ++x;
    printf("x on the process with rank %d: %d\n", rank, x);

    if (rank != (size - 1)) {
        MPI_Send(&x, 1, MPI_INT,
                 rank + 1, 0,
                 MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

