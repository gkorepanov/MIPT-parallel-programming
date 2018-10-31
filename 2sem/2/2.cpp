#include "aux.h"

int main(int argc, char **argv) {
    // auto [N, elems] = parse_args(argc, argv); // doesn't work with #pragma. Why?!
    int N, elems; tie (N, elems) = parse_args(argc, argv);
    omp_set_num_threads(N);

    long double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= elems; i++)
        sum += 1.0L / i;
    cout << setprecision(25) << sum << endl;
}

