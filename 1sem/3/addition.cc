#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <cassert>

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

lint master_sum(lint A, lint B) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // we are master and do not work
    assert(size > 1);
    --size;

    // size of numbers to add
    int num_size = std::max(A.size(), B.size());
    A.resize(num_size, 0);
    B.resize(num_size, 0);

    // num of elems to send to each process
    int elems = num_size / size;
    
    // send number of elements per thread
    MPI_Bcast(&elems, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // scatter parts of a and b
    for (auto i = 0; i < size; ++i) {
        MPI_Send(&A[0] + elems * i, elems,
                 MPI_CHAR, i + 1, 0, MPI_COMM_WORLD);
        MPI_Send(&B[0] + elems * i, elems,
                 MPI_CHAR, i + 1, 0, MPI_COMM_WORLD);
    }

    BYTE carry = 0;
    lint result;
    auto buf = new BYTE[elems + 1];

    // obtain numbers from slaves
    for (auto i = 0; i < size; ++i) {
        MPI_Send(&carry,      1, MPI_CHAR, i + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buf, elems + 1, MPI_CHAR, i + 1, 0, MPI_COMM_WORLD, nullptr);
        result.insert(result.end(), buf, buf + elems);
        carry = buf[elems];
    }

    // calculate the small part left due to (num_size % size) != 0
    auto elems_left = num_size - elems * size;
    if (elems_left) {
        lint a(A.end() - elems_left, A.end());
        lint b(B.end() - elems_left, B.end());
        
        auto sum_left = _sum(a, b, elems_left, carry);
        result.insert(result.end(), sum_left.begin(), sum_left.end());
    }

    // remove trailing zeros
    while ( !result.empty() && result.back() == 0 )
        result.pop_back();

    return result;
}

void slave_sum() {
    int elems;
    // get the number of elements to receive
    MPI_Bcast(&elems, 1, MPI_INT, 0, MPI_COMM_WORLD);
    auto a = new BYTE[elems];
    auto b = new BYTE[elems];

    // obtain numbers from master
    MPI_Recv(a, elems, MPI_CHAR, 0, 0, MPI_COMM_WORLD, nullptr); 
    MPI_Recv(b, elems, MPI_CHAR, 0, 0, MPI_COMM_WORLD, nullptr); 

    lint A(a, a + elems);
    lint B(b, b + elems);

    lint C0 = _sum(A, B, elems, 0);
    lint C1 = _sum(A, B, elems, 1);

    // return sum to master
    BYTE carry;
    MPI_Recv(&carry, 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD, nullptr); 

    if (carry)
        MPI_Send(&C1[0], elems + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD); 
    else
        MPI_Send(&C0[0], elems + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD); 
}


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::ifstream f("input.txt");
        std::string line;

        std::vector<lint> numbers;
        while (std::getline(f, line))
            numbers.push_back(to_lint(line));

        f.close();

        auto result = master_sum(numbers[0], numbers[1]);
        std::cout << to_string(result);
    } else
        slave_sum();

    MPI_Finalize();
    return 0;
}

