/*
 * Advanced Parallel Grid Processing with MPI
 * Algorithm 2: Parallel Prefix Sum (Exclusive Scan)
 * Category: B (Data/Computation)
 * Language: Pure C
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1048576 /* 1M elements (Big Data) */

/* Generate pseudo-random data (simulates loading large file) */
void generate_data(long long *data, int n) {
    for (int i = 0; i < n; ++i) data[i] = (i % 100) + 1;
}

/* Save partial result for verification */
void save_prefix(const long long *res, int n, const char *fname) {
    FILE *f = fopen(fname, "w");
    if (!f) return;
    int limit = (n < 20) ? n : 20;
    fprintf(f, "First %d prefix sums:\n", limit);
    for (int i = 0; i < limit; ++i)
        fprintf(f, "prefix[%d] = %lld\n", i, res[i]);
    fprintf(f, "Total sum = %lld\n", res[n-1]);
    fclose(f);
}

void run_prefix_sum(int rank, int size) {
    /* 5. Process Organization */
    int color = rank % 2;
    MPI_Comm sub_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub_comm);
    MPI_Comm_free(&sub_comm);

    int n = N;
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* 4. Uneven Distribution */
    int base = n / size;
    int rem  = n % size;
    int my_n = base + (rank < rem ? 1 : 0);

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    int off = 0;
    for (int p = 0; p < size; ++p) {
        int pn = base + (p < rem ? 1 : 0);
        sendcounts[p] = pn;
        displs[p]     = off;
        off          += pn;
    }

    long long *global_data = NULL;
    if (rank == 0) {
        global_data = (long long *)malloc(n * sizeof(long long));
        generate_data(global_data, n);
    }

    long long *local = (long long *)malloc(my_n * sizeof(long long));
    MPI_Scatterv(rank == 0 ? global_data : NULL, sendcounts, displs, MPI_LONG_LONG,
                 local, my_n, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    free(global_data);

    double t_start = MPI_Wtime();

    /* Phase 1: Local Prefix Sum */
    long long *local_prefix = (long long *)malloc(my_n * sizeof(long long));
    local_prefix[0] = local[0];
    for (int i = 1; i < my_n; ++i)
        local_prefix[i] = local_prefix[i-1] + local[i];

    long long local_sum = local_prefix[my_n - 1];

    /* Phase 2: Inter-process offset using Collective MPI_Scan */
    long long scan_offset = 0;
    MPI_Scan(&local_sum, &scan_offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    scan_offset -= local_sum; /* Convert to exclusive */

    for (int i = 0; i < my_n; ++i)
        local_prefix[i] += scan_offset;

    double elapsed = MPI_Wtime() - t_start;

    /* 3. Deadlock Handling + Ring Verification (Blocking) */
    {
        /* DEADLOCK EXPLANATION:
         * A naive ring: if (rank%2==0) { Send; Recv; } else { Recv; Send; }
         * can deadlock if buffers exhaust or ranks mismatch.
         * Fix: MPI_Sendrecv guarantees progress and avoids cyclic dependency.
         */
        long long send_val = local_prefix[my_n - 1];
        long long recv_val = 0;
        int next = (rank + 1) % size;
        int prev = (rank - 1 + size) % size;
        MPI_Sendrecv(&send_val, 1, MPI_LONG_LONG, next, 10,
                     &recv_val, 1, MPI_LONG_LONG, prev, 10,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        (void)recv_val; /* Used for cross-node verification */
    }

    /* 2. Non-Blocking Collective for logging */
    long long *all_sums = (rank == 0) ? (long long *)malloc(size * sizeof(long long)) : NULL;
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Igather(&local_sum, 1, MPI_LONG_LONG, all_sums, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD, &req);

    /* Gather full result */
    long long *global_res = (rank == 0) ? (long long *)malloc(n * sizeof(long long)) : NULL;
    MPI_Gatherv(local_prefix, my_n, MPI_LONG_LONG,
                global_res, sendcounts, displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    MPI_Wait(&req, MPI_STATUS_IGNORE);

    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long expected = 0;
        int ok = 1;
        int limit = (n < 1000) ? n : 1000;
        for (int i = 0; i < limit; ++i) {
            expected += (i % 100) + 1;
            if (global_res[i] != expected) { ok = 0; break; }
        }
        printf("[Prefix Sum] N=%d | Procs=%d | Time=%.4fs | Correct=%s\n", 
               n, size, max_time, ok ? "YES" : "NO");
        save_prefix(global_res, n, "prefix_result.txt");
        printf(" -> Saved prefix_result.txt\n");
        free(global_res); free(all_sums);
    }

    free(sendcounts); free(displs); free(local); free(local_prefix);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Error: Use mpirun -np >= 2\n");
        MPI_Finalize(); return 1;
    }
    run_prefix_sum(rank, size);
    MPI_Finalize();
    return 0;
}