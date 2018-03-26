#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank != 0) {
        MPI_Recv(NULL, 0, MPI_BYTE,
                rank - 1, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
    }

    printf("I'm a baby process, my id is %d\n", rank);

    if (rank != (size - 1)) {
        MPI_Send(NULL, 0, MPI_BYTE,
                rank + 1, 0,
                MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

