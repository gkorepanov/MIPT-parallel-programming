#include "aux.h"

int main(int argc, char **argv) {
    auto N = get_num_threads(argc, argv);
    omp_set_num_threads(N);

    int seq = 0;
    #pragma omp parallel shared(seq)
    {
#pragma omp critical
        while ( (seq < N) && (seq != omp_get_thread_num()) );
        
        cout << "Hello world from process " << seq << "!" << endl;
        seq++;
    }
}

