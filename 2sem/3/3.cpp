#include "aux.h"

int main(int argc, char **argv) {
    int N, elems; tie (N, elems) = parse_args(argc, argv);
    omp_set_num_threads(N);

    long double sum = 0;
    vector<long double> min_members(N, 1.0L), part_sum(N, .0L);

    #pragma omp parallel for
    for (int i = 1; i <= elems; i++) {
        auto id = omp_get_thread_num();
        min_members[id] /= i;
        part_sum[id] += min_members[id];
    }
 
    for (uint id = 1; id < N; ++id) {
        min_members[id] *= min_members[id-1];
        part_sum[id] *= min_members[id-1];
    }

    cout << setprecision(25)
         << 1 + accumulate(part_sum.rbegin(),
                           part_sum.rend(), .0L) << endl;
}

