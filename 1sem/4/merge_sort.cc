#include <iostream>
#include <chrono>
#include <fstream>
#include <vector>
#include <pthread.h>
#include <cmath>
#include <cassert>

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

struct Chunk {
    Array::iterator begin;
    Array::iterator end;
};

std::vector<Chunk> split(Array& v, unsigned int num) {
    assert(num < v.size());
    std::vector<Chunk> chunks;
    auto avg = static_cast<float>(v.size()) / num;
    auto last = 0.0;

    while (round(last) < v.size()) {
        chunks.push_back({v.begin() + round(last),
        std::min(v.end(), v.begin() + round(last + avg))});
        last += avg;
    }
    
    return chunks;
}

void* exchange(void *_chunk) {
    auto chunk = static_cast<Chunk*>(_chunk);
    std::sort(chunk->begin, chunk->end);
    return NULL;
}

// parallel sort array inplace
void parallel_sort(Array& array, unsigned int n) {
    auto num_chunks = 2 * n;
    std::vector<pthread_t> tid(n);

    auto chunks = split(array, num_chunks);

    for (size_t i = 0; i < num_chunks; ++i) {
        auto j = i % 2;
        auto it = tid.begin();
        for (; (j + 1) < num_chunks; j += 2, ++it) {
            auto chunk = new Chunk {chunks[ j ].begin, chunks[j+1].end};
            pthread_create(&(*it), 0, exchange, chunk);
        }

        j = i % 2; it = tid.begin();
        for (; (j + 1) < num_chunks; j += 2, ++it)
            pthread_join(*it, NULL);

    }
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

