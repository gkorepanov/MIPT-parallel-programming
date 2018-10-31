#include <omp.h>
#include <iostream>
#include <string>

using namespace std;

int get_num_threads(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage: <" << argv[0]
             << "> <number of threads>" << endl;
        exit(1);
    }

    return stoi(argv[1]);
}

