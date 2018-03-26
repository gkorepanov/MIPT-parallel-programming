#include <stdio.h>
#include <mpi.h>
#include <math.h>

void Bcast_time()
{
	int rank = 0;
	double start = 0, stop = 0, tick = MPI_Wtick();
	double prev_avg = 0, avg = 0, curr = 0, diff = 0;
	int terminate = 0;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int iter = 1;

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		start = MPI_Wtime();
	}

	MPI_Bcast(&terminate, sizeof(int), MPI_CHAR, 0, MPI_COMM_WORLD);
		
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		avg = curr = MPI_Wtime() - start;
	}
	
	while(!terminate) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			++iter;
			start = MPI_Wtime();
		}

		/*Bcast itself*/
		if(MPI_SUCCESS != MPI_Bcast(&terminate, sizeof(int), MPI_CHAR, 0, MPI_COMM_WORLD)) {
			printf("Error occured in Bcast\n");
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			curr = MPI_Wtime() - start;
            printf("TIME: %.20lf\n", curr);
			avg = (avg * iter + curr) / (iter + 1);
			diff = fabs(avg - prev_avg);
			if (diff < tick) {
				++terminate;
			}
			if (!(iter % 200 )) {
				printf("Iter %d; Avg %.9f; diff %g\n", 
					    iter, avg, diff);
			}
			prev_avg = avg;
		}

		if(MPI_SUCCESS != MPI_Bcast(&terminate, sizeof(int), MPI_CHAR, 0, MPI_COMM_WORLD)) {
			printf("Error occured in Bcast\n");
		}
	}

	if (rank == 0) {
		printf("Bcast;   time %.9f with error %g on iteration %d \n", avg, diff, iter);
	}
	return;
}

void Reduce_time()
{
	int rank = 0;
	double start = 0, stop = 0, tick = MPI_Wtick();
	double prev_avg = 0, avg = 0, curr = 0, diff = 0;
	int terminate = 0;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int local_sum = rank;
	int global_sum = 0;

	int iter = 1;

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		start = MPI_Wtime();
	}

	/* First iteration handled separatly because
	 * accuracy cannot be defined yet.
	 */
	MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM,
		       0, MPI_COMM_WORLD);
		
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		avg = curr = MPI_Wtime() - start;
	}
	
	while(!terminate) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			++iter;
			start = MPI_Wtime();
		}

		MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM,
		       0, MPI_COMM_WORLD);
		
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			stop = MPI_Wtime();
			curr = stop - start;
			avg = (avg * iter + curr) / (iter + 1);
			diff = fabs(avg - prev_avg);
			if (diff < tick) {
				++terminate;
			}
			if (!(iter % 200 )) {
				printf("Iter %d; Avg %.9f; diff %.9f \n", iter, avg, diff);
			}
			prev_avg = avg;
		}

		//MPI_Barrier(MPI_COMM_WORLD);
		if(MPI_SUCCESS != MPI_Bcast(&terminate, sizeof(int), MPI_CHAR, 0, MPI_COMM_WORLD)) {
			printf("Error occured in Bcast\n");
		}
	}

	//printf("dumb exit %d", rank);
	if (rank == 0) {
		printf("Reduce;  time %.9f with error %g on iteration %d \n", avg, diff, iter);
	}
	return;
}

void Scatter_time()
{
	int rank = 0, size = 0;
	double start = 0, stop = 0, tick = MPI_Wtick();
	double prev_avg = 0, avg = 0, curr = 0, diff = 0;
	int terminate = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int array[size], local = 0;

	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int iter = 1;

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		start = MPI_Wtime();
		
		for (int i = 0; i < size; ++i) {
			array[i] = i;
		}
	}

	MPI_Scatter(array, 1, MPI_INT, &local, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		avg = curr = MPI_Wtime() - start;
	}
	
	while(!terminate) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			++iter;
			start = MPI_Wtime();
		}

		MPI_Scatter(array, 1, MPI_INT, &local, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			stop = MPI_Wtime();
			curr = stop - start;
			avg = (avg * iter + curr) / (iter + 1);
			diff = fabs(avg - prev_avg);
			if (diff < tick) {
				++terminate;
			}
			if (!(iter % 200 )) {
				printf("Iter %d; Avg %.9f; diff %.9f\n", iter, avg, diff);
			}
			prev_avg = avg;
		}

		//MPI_Barrier(MPI_COMM_WORLD);
		if(MPI_SUCCESS != MPI_Bcast(&terminate, sizeof(int), MPI_CHAR, 0, MPI_COMM_WORLD)) {
			printf("Error occured in Bcast\n");
		}
	}

	//printf("dumb exit %d", rank);
	if (rank == 0) {
		printf("Scatter; time %.9f with error %g on iteration %d \n", avg, diff, iter);
	}
	return;
}

void Gather_time()
{
	int rank = 0, size = 0;
	double start = 0, stop = 0, tick = MPI_Wtick();
	double prev_avg = 0, avg = 0, curr = 0, diff = 0;
	int terminate = 0;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	int array[size], local = rank;
	int iter = 1;

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		start = MPI_Wtime();
	}

	MPI_Gather(&local, 1, MPI_INT, array, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0) {
		avg = curr = MPI_Wtime() - start;
	}
	
	while(!terminate) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			++iter;
			start = MPI_Wtime();
		}

		MPI_Gather(&local, 1, MPI_INT, array, 1, MPI_INT, 0, MPI_COMM_WORLD);
		
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0) {
			stop = MPI_Wtime();
			curr = stop - start;
			avg = (avg * iter + curr) / (iter + 1);
			diff = fabs(avg - prev_avg);
			/*if (diff > max_diff) {
				max_diff = diff;
			}*/
			if (diff < tick) {
				++terminate;
			}
			if (!(iter % 200 )) {
				printf("Iter %d; Avg %.9f; diff %.9f\n", iter, avg, diff);
			}
			prev_avg = avg;
		}

		//MPI_Barrier(MPI_COMM_WORLD);
		if(MPI_SUCCESS != MPI_Bcast(&terminate, sizeof(int), MPI_CHAR, 0, MPI_COMM_WORLD)) {
			printf("Error occured in Bcast\n");
		}
	}

	if (rank == 0) {
		printf("Gather;  time %.9f with error %g on iteration %d \n", avg, diff, iter);
	}
	return;
}

int main (int argc, char** argv)
{
	int rank = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (!rank) {
		printf("Wtick is %g\n", MPI_Wtick());		
	}
	Bcast_time();
	Reduce_time();
	Scatter_time();
	Gather_time();
	MPI_Finalize();

	return 0;
}
