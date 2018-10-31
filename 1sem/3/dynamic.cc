#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cassert>
#include <list>

#include <mpi.h>

typedef char BYTE;
typedef std::vector<BYTE> lint;

std::string to_string(lint num) {
    std::string res;
    for (auto &elem : num)
        res = std::string({
            static_cast<char>('0' + elem / 10),
            static_cast<char>('0' + elem % 10)
        }) + res;

    if (res.front() == '0')
        res.erase(res.begin());

    return res;
}

lint to_lint(std::string str) {
    lint res;
    auto i = 0;
    for (auto it = str.rbegin(); it != str.rend(); ++it, ++i) {
        if (i % 2) res.back() += (*it - '0') * 10;
        else       res.push_back (*it - '0');
    }
    return res;
}

lint _sum(lint a, lint b, size_t size, BYTE carry) {
    assert(a.size() == size);
    assert(b.size() == size);

    lint result(size + 1, 0);
    a[0] += carry;

    for (auto i = 0; i < size; ++i) {
        auto sum = a[i] + b[i];
        result[i]   += sum % 100;
        result[i+1] += sum / 100; 
    }

    return result;
}

struct Chunk {
    BYTE* begin;
    int elems;
};

std::vector<Chunk> split(lint& v, size_t elems) {
    assert(elems <= v.size());
    std::vector<Chunk> chunks;

    for (auto it = v.begin(); it < v.end(); it += elems)
        chunks.push_back({&(*it), static_cast<int>(std::min(it + elems, v.end()) - it)});

    return chunks;
}

lint master_sum(lint A, lint B, size_t elems) {
    std::ofstream of("master.txt");
    of << "MASTER: " << std::endl;

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // we are master and do not work
    assert(size > 1);
    --size;

    // size of numbers to add
    int num_size = std::max(A.size(), B.size());
    A.resize(num_size, 0);
    B.resize(num_size, 0);

    auto chunks_A = split(A, elems);
    auto chunks_B = split(B, elems);

    of << "elems: " << elems << std::endl;
    of << "slaves: " << size << std::endl;

    std::list<int> slaves;

    for (int i = 0; i < size; ++i)
        slaves.push_back(i);

    std::vector<MPI_Request> requests_0(size, MPI_REQUEST_NULL), requests_1(size, MPI_REQUEST_NULL);
    std::vector<lint> results_0, results_1;

    int index;
    for (int i = 0; i < chunks_A.size(); ++i) {
        auto chunk_A = chunks_A[i];
        auto chunk_B = chunks_B[i];
        auto elems = chunk_A.elems;

        // if no ready slave, wait for some
        if (slaves.empty()) {
            of << "waiting some slave... " << std::endl;
            MPI_Waitany(requests_0.size(), &requests_0[0], &index, MPI_STATUS_IGNORE);
            of << "slave finished:  " << index << std::endl;
            of << "waiting this slave with carry == 1... " << std::endl;
            MPI_Wait(&requests_1[index], MPI_STATUS_IGNORE);
            of << "OK! " << std::endl;
            slaves.push_back(index);
        }

        // get random (last) ready slave
        auto slave = slaves.back();
        slaves.pop_back();

        of << "sending elems " << elems << " to slave " << slave << std::endl;
        // send info anount number of elems
        MPI_Send(&chunk_A.elems, 1,
                 MPI_INT, slave, 0, MPI_COMM_WORLD);

        // send elems themselves
        MPI_Send(chunk_A.begin, elems,
                 MPI_CHAR, slave, 0, MPI_COMM_WORLD); // A

        MPI_Send(chunk_B.begin, elems,
                 MPI_CHAR, slave, 0, MPI_COMM_WORLD); // B

        of << "sent elems " << elems << " to slave " << slave << std::endl;

        // obtain results in non-blocking mode
        results_0.push_back(lint(elems + 1));
        MPI_Irecv(&results_0.back()[0], elems + 1,
                 MPI_CHAR, slave, 0, MPI_COMM_WORLD, &requests_0[slave]); // carry == 0

        results_1.push_back(lint(elems + 1));
        MPI_Irecv(&results_1.back()[0], elems + 1,
                 MPI_CHAR, slave, 1, MPI_COMM_WORLD, &requests_1[slave]); // carry == 1
        
        of << "sent non-blocking read requests to slave " << slave << std::endl;
    }

    MPI_Waitall(requests_0.size(), &requests_0[0], MPI_STATUSES_IGNORE);
    MPI_Waitall(requests_1.size(), &requests_1[0], MPI_STATUSES_IGNORE);

    of << "everybod finished, cpollecting results... " << std::endl;

    // send FIN to slaves

    auto fin = -1;
    for (int i = 0; i < size; ++i)
        MPI_Send(&fin, 1, MPI_INT, i, 0, MPI_COMM_WORLD);


    // calculate the final number

    BYTE carry = 0;
    lint result, cur;

    for (auto i = 0; i < chunks_A.size(); ++i) {
        auto cur = carry ? results_1[i] : results_0[i];

        result.insert(result.end(), cur.begin(), cur.end() - 1);
        carry = cur.back();
    }

    // remove trailing zeros
    while ( !result.empty() && result.back() == 0 )
        result.pop_back();

    return result;
}

void slave_sum(int master) {
    while (true) {
        int elems;

        // obtain number of elements
        MPI_Recv(&elems, 1, MPI_INT, master, 0, MPI_COMM_WORLD, nullptr); 

        if (elems == -1) // FIN signal
            break;

        lint A(elems), B(elems);

        // obtain numbers from master
        MPI_Recv(&A[0], elems, MPI_CHAR, master, 0, MPI_COMM_WORLD, nullptr); 
        MPI_Recv(&B[0], elems, MPI_CHAR, master, 0, MPI_COMM_WORLD, nullptr); 

        lint C0 = _sum(A, B, elems, 0);
        lint C1 = _sum(A, B, elems, 1);

        // return sums to master
        MPI_Send(&C0[0], elems + 1, MPI_CHAR, master, 0, MPI_COMM_WORLD); // carry == 0
        MPI_Send(&C1[0], elems + 1, MPI_CHAR, master, 1, MPI_COMM_WORLD); // carry == 1
    }
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto master = size - 1; // master is the last

    if (rank == master) {
        std::ifstream f("input.txt");
        std::string line;

        std::vector<lint> numbers;
        while (std::getline(f, line))
            numbers.push_back(to_lint(line));

        f.close();

        std::cout << to_string(numbers[0]) << std::endl;
        std::cout << to_string(numbers[1]) << std::endl;
        auto result = master_sum(numbers[0], numbers[1], 2);
        std::cout << to_string(result);
    } else
        slave_sum(master);

    MPI_Finalize();
    return 0;
}

