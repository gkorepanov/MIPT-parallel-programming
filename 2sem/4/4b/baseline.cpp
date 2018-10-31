#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>

#define ISIZE 1000
#define JSIZE 1000
#define SIZE ISIZE * JSIZE

using namespace std;

double f(double x) {
    return sin(0.00001 * x);
}

void calculate(double a[ISIZE][JSIZE], int id, int nthreads) {
    for (size_t i = 3; i < ISIZE; ++i)
        for (size_t j = 0; j < JSIZE - 2; ++j)
            a[i][j] = f( a[i-3][j+2] );
}


void output(double a[ISIZE][JSIZE]) {
    FILE *ff;
    ff = fopen("result-baseline.txt", "w");

    for(int i = 0; i < ISIZE; i++) {
        for (int j = 0; j < JSIZE; j++)
            fprintf(ff, "%f ", a[i][j]);

        fprintf(ff,"\n");
    }

    fclose(ff);
}


double a[ISIZE][JSIZE];

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int id, nthreads;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &nthreads);

    int i, j;

    if (id == 0)
        for (i = 0; i < ISIZE; i++)
            for (j = 0; j < JSIZE; j++)
                a[i][j] = 10 * i + j;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = MPI_Wtime();

    calculate(a, id, nthreads);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = MPI_Wtime();

    auto time = end_time - start_time;
    
    if (id == 0) {
        cout << std::setprecision(20) << time << std::endl;
        output(a);
    }

    MPI_Finalize();
}
