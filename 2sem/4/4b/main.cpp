#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <algorithm>

#define ISIZE 1000
#define JSIZE 1000
#define SIZE ISIZE * JSIZE

using namespace std;

double f(double x) {
    return sin(0.00001 * x);
}

#define ISTEP -3
#define JSTEP +2

// parallel equivalient of a[i][j] = f( a[i-3][j+2] );
void calculate(double _a[ISIZE][JSIZE], int id, int nthreads) {
    assert(ISTEP < 0);
    double *a = (double*) _a;

    // _PT == PER THREAD
    size_t processed_elements_per_iteration = JSIZE - abs(JSTEP);
    size_t processed_elements_PT = processed_elements_per_iteration / nthreads;

    // SEND VECTORS
    vector<int> send_starts(nthreads);
    for (size_t k = 0; k < nthreads; ++k)
        send_starts[k] = max(JSTEP, 0) + processed_elements_PT * k;

    vector<int> send_counts(nthreads, processed_elements_PT);
    // truncate last to prevent overflow
    send_counts.back() = processed_elements_per_iteration \
                       - send_starts.back() + max(JSTEP, 0); 

    // RECEIVE VECTORS
    auto receive_starts = send_starts;

    for (auto& elem : receive_starts)
        elem += -JSIZE*ISTEP - JSTEP;
    
    auto& receive_counts = send_counts;

    // PT VECTOR
    vector<double> buf(processed_elements_PT + (nthreads - 1));

    for (size_t i = max(0, -ISTEP); i < min(ISIZE, ISIZE - ISTEP); ++i) {

        // #############################################################
        // #                         SCATTER                           #
        // #############################################################

        MPI_Scatterv(a, send_counts.data(), send_starts.data(), MPI_DOUBLE,
                     buf.data(), send_counts[id],
                     MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // #############################################################
        // #                         PROCESS                           #
        // #############################################################

        for (size_t j = 0; j < send_counts[id]; ++j)
            buf[j] = f( buf[j] );


        // #############################################################
        // #                         GATHER                            #
        // #############################################################
        
        MPI_Gatherv(buf.data(), receive_counts[id], MPI_DOUBLE,
                    a, receive_counts.data(), receive_starts.data(), MPI_DOUBLE,
                    0, MPI_COMM_WORLD);


        // #############################################################
        // #                    POINTERS UPDATE                        #
        // #############################################################
        
        for (auto& elem : send_starts)
            elem += JSIZE;
        for (auto& elem : receive_starts)
            elem += JSIZE;
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

