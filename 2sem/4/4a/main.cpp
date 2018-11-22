#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <algorithm>

#define ISIZE 10000
#define JSIZE 10000
#define SIZE ISIZE * JSIZE

using namespace std;

double f(double x) {
    return sin(0.00001 * x);
}

#define ISTEP 3
#define JSTEP -2

// parallel equivalient of a[i][j] = f( a[i+3][j-2] );
void calculate(double _a[ISIZE][JSIZE], int id, int nthreads) {
    assert(ISTEP > 0);
    assert(JSTEP < 0);
    double *a = (double*) _a;

    // _PT == PER THREAD
    size_t processed_lines_PT = ISIZE / nthreads;
    size_t total_lines_PT = processed_lines_PT + ISTEP;

    size_t processed_elements_PT = processed_lines_PT * JSIZE;
    size_t total_elements_PT     = total_lines_PT     * JSIZE;


    // #############################################################
    // #                         SCATTER                           #
    // #############################################################
    vector<int> send_starts(nthreads);
    for (size_t i = 0; i < nthreads; ++i)
        send_starts[i] = processed_elements_PT * i;

    vector<int> send_counts(nthreads, total_elements_PT);
    // truncate last to prevent overflow
    send_counts.back() = SIZE - send_starts.back();

    vector<double> buf(total_elements_PT + (nthreads - 1));

    MPI_Scatterv(a, send_counts.data(), send_starts.data(), MPI_DOUBLE,
                 buf.data(), send_counts[id],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // #############################################################
    // #                         PROCESS                           #
    // #############################################################
    for (size_t i = 0; i < send_counts[id] / JSIZE - ISTEP; ++i)
        for (size_t j = max(0, -JSTEP); j < min(JSIZE, JSIZE - JSTEP); ++j)
            buf[i*JSIZE + j] = f( buf[ (i+ISTEP)*JSIZE + (j+JSTEP) ] );


    // #############################################################
    // #                         GATHER                            #
    // #############################################################
    auto& receive_starts = send_starts;

    vector<int> receive_counts(nthreads, processed_elements_PT);
    // truncate last to prevent overflow
    receive_counts.back() = SIZE - receive_starts.back();
    
    MPI_Gatherv(buf.data(), receive_counts[id], MPI_DOUBLE,
                a, receive_counts.data(), receive_starts.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
}


void output(double a[ISIZE][JSIZE]) {
 //    FILE *ff;
 //    ff = fopen("result.txt", "w");
 //
 //    for(int i = 0; i < ISIZE; i++) {
 //        for (int j = 0; j < JSIZE; j++)
 //            fprintf(ff, "%f ", a[i][j]);
 //
 //        fprintf(ff,"\n");
 //    }
 //
 //    fclose(ff);
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

