#include <omp.h>
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <numeric>
#include <string>
#include <functional>

using namespace std;

auto parse_args(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Usage: <" << argv[0]
             << "> <number of threads> <numver of elements>" << endl;
        exit(1);
    }

    return tuple {stoi(argv[1]), stoi(argv[2])};
}

