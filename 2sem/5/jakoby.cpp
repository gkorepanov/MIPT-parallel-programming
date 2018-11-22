#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <algorithm>

#define N 100
#define ISIZE N
#define JSIZE N
#define SIZE ISIZE * JSIZE

using namespace std;

// parallel equivalient of a[i][j] = f( a[i-1][j] + a[i][j-1] + a[i][j+1] + a[i+1][j] );
void calculate(double _a[ISIZE][JSIZE], int id, int nthreads) {
    double *a = (double*) _a;

    // _PT == PER THREAD
    size_t processed_lines_PT = (ISIZE - 2) / nthreads;
    size_t total_lines_PT = processed_lines_PT + 2;

    size_t processed_elements_PT = processed_lines_PT * JSIZE;
    size_t total_elements_PT     = total_lines_PT     * JSIZE;

    vector<double> buf(total_elements_PT + (nthreads - 1));

    // #############################################################
    // #                         SCATTER                           #
    // #############################################################
    vector<int> send_starts(nthreads);
    for (size_t i = 0; i < nthreads; ++i)
        send_starts[i] = processed_elements_PT * i;

    vector<int> send_counts(nthreads, total_elements_PT);
    // truncate last to prevent overflow
    send_counts.back() = SIZE - send_starts.back();

    // #############################################################
    // #                         GATHER                            #
    // #############################################################

    auto receive_starts = send_starts;
    for (size_t i = 0; i < nthreads; ++i)
        receive_starts[i] += JSIZE;

    vector<int> receive_counts(nthreads, processed_elements_PT);
    // truncate last to prevent overflow
    receive_counts.back() = (SIZE - JSIZE) - receive_starts.back();

    int calculate = true;
    double residual_pt, residual;

    // #############################################################
    // #                         PROCESS                           #
    // #############################################################
    uint iter = 0;
    for(iter = 0; calculate; ++iter) {
        residual_pt = 0;

        MPI_Scatterv(a, send_counts.data(), send_starts.data(), MPI_DOUBLE,
                     buf.data(), send_counts[id],
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);


        for (size_t i = 1; i < receive_counts[id] / JSIZE + 1; ++i)
            for (size_t j = 1; j < N - 1; ++j) {
                auto prev = buf[i*JSIZE + j];

                buf[i*JSIZE + j] = (buf[ (i-1)*JSIZE + ( j ) ] + buf[ ( i )*JSIZE + (j-1) ] + \
                                    buf[ ( i )*JSIZE + (j+1) ] + buf[ (i+1)*JSIZE + ( j ) ]) / 4;

                residual_pt += abs(prev - buf[i*JSIZE + j]);
            }

        
        MPI_Gatherv(buf.data() + JSIZE, receive_counts[id], MPI_DOUBLE,
                    a, receive_counts.data(), receive_starts.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        MPI_Reduce(&residual_pt, &residual, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 

        if (id == 0) 
            calculate = (residual > 0.001);

        MPI_Bcast(&calculate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (id == 0) {
        cout << "iterations: " << iter << endl;
    }

}


void output(double a[ISIZE][JSIZE]) {
    FILE *ff;
    ff = fopen("result.txt", "w");

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
    double _average = (100. + 200. + 300. + 400.) / 4;

    if (id == 0)
        for (i = 0; i < ISIZE; i++)
            for (j = 0; j < JSIZE; j++)
                a[i][j] = _average;

    for (uint j = 0; j < N; ++j) {
        a[  j  ][  0  ] = 100;
        a[  0  ][  j  ] = 200;
        a[  j  ][ N-1 ] = 300;
        a[ N-1 ][  j  ] = 400;
    }

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

