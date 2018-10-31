#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <pthread.h>
#include <cmath>
#include <cassert>
#include "barrier.h"
#include <unistd.h>
#include <string>

typedef long long int lint;
typedef std::vector<lint> Array;

Array read_array() {
    std::ifstream infile("input.txt");
    Array array;

    lint num;
    while (infile >> num)
        array.push_back(num);

    infile.close();
    return array;
}   

inline void print_array(const Array& array) {
    for (auto &elem : array)
        std::cout << elem << " ";
    std::cout << std::endl;
}

struct Data {
    pthread_barrier_t* barrier;
    Array& array;
    const size_t n;
    const size_t i;
};

void cmpexchange(lint& a, lint& b) {
    if (a > b)
        std::swap(a, b);
}

void* thread_sort(void* _data) {
    auto *data = static_cast<Data*>(_data);
    auto size = (data->array).size();

    for (size_t t = 0; t < size; ++t) {
        size_t npairs = (size - (t % 2)) / 2;
        auto   avg    = static_cast<float> (npairs) / data->n;
        size_t start  = (t % 2) + 2*round(avg * (data->i));
        size_t end    = (t % 2) + 2*round(avg * (data->i + 1));
        end = std::min(end, size - 1);

        for (auto j = start; j < end; j += 2)
            cmpexchange(data->array[j], data->array[j+1]);

        pthread_barrier_wait(data->barrier);
    }

    return NULL;
}

// parallel sort array inplace
void parallel_sort(Array& array, const size_t n) {
    std::vector<pthread_t> tids(n);
    std::vector<Data> data;

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, n);

    for (size_t i = 0; i < n; ++i) {
        data.push_back(Data {&barrier, array, n, i});
        pthread_create(&tids[i], 0, thread_sort, &(data.back()));
    }

    for (auto &tid : tids)
        pthread_join(tid, NULL);

    pthread_barrier_destroy(&barrier);

    for (size_t i = 0; i < n; ++i)
        std::cout << data[i].barrier << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [n] " << std::endl;
        return -1;
    }

    unsigned int n = std::stoi(argv[1]);
    auto array = read_array();
    print_array(array);

    auto start = std::chrono::steady_clock::now();
    parallel_sort(array, n);
    auto end = std::chrono::steady_clock::now();

    print_array(array);
    auto duration = std::chrono::duration <double, std::milli> (end - start).count();

    std::cout << "Time taken to sort:" << std::endl;
    std::cout << "\t" << duration << " ms" << std::endl;
}

