#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>

#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <cassert>

struct Time {
    double value;
    double error;
};

struct Func {
    void (*ptr)(std::string);
    char name[32];
};

int rank, size;

void bcast   (std::string type);
void reduce  (std::string type);
void gather  (std::string type);
void scatter (std::string type);

std::vector<std::string> types = {"MPI", "Custom"};

Func funcs[] = {
    (Func) { bcast,   "BCast"   },
    (Func) { reduce,  "Reduce"  },
    (Func) { scatter, "Scatter" },
    (Func) { gather,  "Gather"  }
};

constexpr int N_funcs = sizeof(funcs)/sizeof(Func);

// Measure execution time of given function run all MPI threads
Time measure_time(std::function<void()> func, std::string name) {
    // File to write distinct times to (for distribution plots)
    std::ofstream f(name);

    constexpr int N = 10000; // number of times to run
    double start_time = 0, end_time = 0;
    double times_sum = 0, squares_sum = 0, time = 0;

    for (int i = 0; i < N; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        func();

        MPI_Barrier(MPI_COMM_WORLD);
        end_time = MPI_Wtime();

        time = end_time - start_time;
        f << std::setprecision(20) << time;
        times_sum += time;

        squares_sum += pow(time, 2);
        f.close();
    }

    // average execution time
    double average_time = times_sum / N;
    // average quadratic error
    double error = sqrt( (squares_sum - pow(time, 2)) / N / (N-1) );

    return (Time) {average_time, error};
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    for (int i = 0; i < N_funcs; ++i) {
        for (auto type : types) {
            auto name = type + "_" + funcs[i].name;
            auto func = [&]() { (funcs[i].ptr)(type); };

            Time time = measure_time(func, name);
            if (rank == 0) {
                printf("%s:\n\t(%.0lf +- %.0lf) ns (%.0lf%% error)\n",
                        name.c_str(), 1e9*time.value,
                        1e9*time.error, 100 * time.error/time.value);
            }
        }
    }

    if (rank == 0)
        printf("\nWtick accuracy: %.1lf ns\n", 1e9 * MPI_Wtick());

    MPI_Finalize();
    return 0;
}



////////////////////////////////////////////////////////////
//     Custom implementation of 4 basic MPI functions     //
////////////////////////////////////////////////////////////

// Errors are not handled, though in C++ exceptions are thrown

int G_Bcast(void* buf, int count, MPI_Datatype type, int root, MPI_Comm comm) {
    MPI_Barrier(comm);
    if (rank == root) {
        for (auto i = 0; i < size; ++i)
            MPI_Send(buf, count, type, i, 0, comm);
    } else {
        MPI_Recv(buf, count, type, root, 0, comm, nullptr);
    }
    MPI_Barrier(comm);
    return 0;
}

// MPI_INT SUM only
int G_Reduce(const void *sendbuf, void *recvbuf, int count,
             MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    MPI_Barrier(comm);

    assert(op       == MPI_SUM);
    assert(datatype == MPI_INT);

    MPI_Send(sendbuf, count, datatype, root, 0, comm);

    if (rank == root) {
        int typesize;
        MPI_Type_size(datatype, &typesize);
        assert(typesize == sizeof(int));

        std::fill((int*)recvbuf, (int*)recvbuf + typesize*count, 0);
        void* tempbuf = calloc(count, typesize);

        for (auto i = 0; i < size; ++i) {
            MPI_Recv(tempbuf, count, datatype, i, 0, comm, nullptr);
            for (auto j = 0; j < count; ++j) {
                *((int*)recvbuf + j)  += *((int*)tempbuf + j);
            }
        }
    }

    MPI_Barrier(comm);
    return 0;
}

int G_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
              void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
              MPI_Comm comm) {
    MPI_Barrier(comm);

    if (rank == root) {
        int typesize;
        MPI_Type_size(sendtype, &typesize);

        for (auto i = 0; i < size; ++i) {
            MPI_Send((char*)sendbuf +  i * sendcount * typesize, sendcount, sendtype, i, 0, comm);
        }
    }

    MPI_Recv(recvbuf, recvcount, recvtype, root, 0, comm, nullptr);

    MPI_Barrier(comm);
    return 0;
}

int G_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
             void *recvbuf, int recvcount, MPI_Datatype recvtype, int root,
             MPI_Comm comm) {
    MPI_Barrier(comm);

    MPI_Send(sendbuf, sendcount, sendtype, root, 0, comm);

    if (rank == root) {
        int typesize;
        MPI_Type_size(sendtype, &typesize);

        for (auto i = 0; i < size; ++i) {
            MPI_Recv((char*)sendbuf +  i * sendcount * typesize, recvcount, recvtype, i, 0, comm, nullptr);
        }
    }

    MPI_Barrier(comm);
    return 0;

}

void bcast(std::string type) {
    auto func = (type == "MPI") ? MPI_Bcast : G_Bcast;
    int buf;
    func(&buf, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void reduce(std::string type) {
    auto func = (type == "MPI") ? MPI_Reduce : G_Reduce;
    int sendbuf;
    int recvbuf;
    func(&sendbuf, &recvbuf, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}

void gather(std::string type) {
    auto func = (type == "MPI") ? MPI_Gather : G_Gather;
    int* rbuf = (int*) ((rank == 0) ? calloc(size, sizeof(int)) : NULL);
    int  sbuf;

    func(&sbuf, 1, MPI_INT, rbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void scatter(std::string type) {
    auto func = (type == "MPI") ? MPI_Scatter : G_Scatter;
    int* sbuf = (int*) ((rank == 0) ? calloc(size, sizeof(int)) : NULL);
    int  rbuf;

    func(sbuf, 1, MPI_INT, &rbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

