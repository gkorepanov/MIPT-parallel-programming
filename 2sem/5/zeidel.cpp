#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <algorithm>
#include <stdexcept>

#define N 100

using namespace std;

typedef vector<double> Diag;

// Matrix keeping elements in vector of diagonal
// vectors instead of vector of rows
class Matrix {
private:
    size_t n;

public:
    vector<Diag> diags;
    double residual;

    inline double& operator() (uint x, uint y) {
        return diags[x + y][x];
    }

    void calculate_diag(uint k);

    inline size_t diag_start(uint k) {
        return (k < n) ? 0 : (k - n + 1);
    }

    inline size_t diag_end(uint k) {
        return (k < n) ? (k + 1) : n;
    }

    void print();

    Matrix(size_t n, double value = 0);
};

Matrix::Matrix(size_t n, double value) :
    n(n),
    residual(0),
    diags(2*n + 1, Diag(n, value))
{}

void Matrix::calculate_diag(uint k) {
    if ((k < 2) || (k > 2*n - 4))
        throw invalid_argument("Wrong diagonal number in calculate_diag()");

    // Zeidel's method
    for (uint j = diag_start(k) + 1; j < diag_end(k) - 1; ++j) {
        auto prev = diags[k][j];

        diags[k][j] = (diags[k-1][j-1] + diags[k-1][ j ] + \
                       diags[k+1][ j ] + diags[k+1][j+1]) / 4 ;

        residual += abs(diags[k][j] - prev);
    }
}

void Matrix::print() {
    for(int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++)
            cout << setw(10) << (*this)(x, y) << " ";
        cout << endl;
    }
}

void send(double* ptr, int dst) {
    MPI_Request request;
    MPI_Isend((void*) ptr, N, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &request);
}

void receive(double* ptr, int src) {
    MPI_Status status;
    MPI_Recv((void*) ptr, N, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &status);
}

Matrix calculate(int id, int nthreads) {
    double _average = (100. + 200. + 300. + 400.) / 4;
    Matrix m(N, _average);

    for (uint j = 0; j < N; ++j) {
        m(  0  ,  j  ) = 100;
        m(  j  ,  0  ) = 200;
        m( N-1 ,  j  ) = 300;
        m(  j  , N-1 ) = 400;
    }

    uint i = id;

    while (true) {
        m.residual = 0;

        for (uint k = 2; k < 2*N - 3; ++k) {
            if ( (k < 2*N - 4) && (i != 0) )
                receive(m.diags[k+1].data(), (id - 1 + nthreads) % nthreads);
            
            m.calculate_diag(k);

            if (k > 2)
                send(m.diags[k].data(), (id + 1) % nthreads);
        }
        
        if ( m.residual < 0.001 )
            break;

        i += nthreads;
    }

    if (id == 0)
        cout << i << " iterations" << endl;

    return m;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int id, nthreads;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &nthreads);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = MPI_Wtime();

    auto m = calculate(id, nthreads);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = MPI_Wtime();

    auto time = end_time - start_time;
    
    if (id == 0) {
        cout << "Time: " << setprecision(20) << time << " s" << std::endl;
        m.print();
    }

    MPI_Finalize();
}
