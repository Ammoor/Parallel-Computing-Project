/*
 * Advanced Parallel Grid Processing with MPI
 * Algorithm 1: Heat Diffusion (5-Point Stencil)
 * Category: A (Grid/Spatial)
 * Language: Pure C
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ROWS 512
#define COLS 512
#define STEPS 500
#define ALPHA 0.1
#define DT 0.25
#define DX 1.0

/* Initialize grid: hot center (100), top border (50), rest (0) */
void init_grid(double *grid, int rows, int cols) {
    memset(grid, 0, rows * cols * sizeof(double));
    int cr = rows / 2, cc = cols / 2;
    int half = rows / 8;
    for (int r = cr - half; r < cr + half; ++r)
        for (int c = cc - half; c < cc + half; ++c)
            grid[r * cols + c] = 100.0;
    for (int c = 0; c < cols; ++c) grid[c] = 50.0; /* Top boundary */
}

/* Save result to CSV */
void save_grid(const double *grid, int rows, int cols, const char *fname) {
    FILE *f = fopen(fname, "w");
    if (!f) return;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c)
            fprintf(f, "%.4f%s", grid[r * cols + c], (c + 1 < cols) ? "," : "\n");
    }
    fclose(f);
}

void run_heat_diffusion(int rank, int size) {
    /* 5. Process Organization (MPI_Comm_split) */
    int color = rank % 2;
    MPI_Comm sub_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &sub_comm);
    /* Use sub_comm for logical grouping if needed, then free */
    MPI_Comm_free(&sub_comm);

    /* Broadcast simulation parameters */
    int params[3] = { ROWS, COLS, STEPS };
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int rows = params[0], cols = params[1], steps = params[2];

    /* 4. Uneven Data Distribution */
    int base = rows / size;
    int rem  = rows % size;
    int my_rows = base + (rank < rem ? 1 : 0);

    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));
    int offset = 0;
    for (int p = 0; p < size; ++p) {
        int pr = base + (p < rem ? 1 : 0);
        sendcounts[p] = pr * cols;
        displs[p]     = offset;
        offset       += pr * cols;
    }

    double *global_grid = NULL;
    if (rank == 0) {
        global_grid = (double *)malloc(rows * cols * sizeof(double));
        init_grid(global_grid, rows, cols);
    }

    /* Local buffer with ghost rows (top + bottom) */
    int local_size = (my_rows + 2) * cols;
    double *local     = (double *)calloc(local_size, sizeof(double));
    double *local_new = (double *)calloc(local_size, sizeof(double));

    /* Distribute initial state */
    MPI_Scatterv(rank == 0 ? global_grid : NULL, sendcounts, displs, MPI_DOUBLE,
                 local + cols, my_rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    double t_start = MPI_Wtime();

    for (int step = 0; step < steps; ++step) {
        /* 2. Communication Strategy: Non-Blocking (Even steps) */
        if (step % 2 == 0) {
            MPI_Request reqs[4];
            MPI_Isend(local + cols,              cols, MPI_DOUBLE, up,   0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Irecv(local,                     cols, MPI_DOUBLE, up,   1, MPI_COMM_WORLD, &reqs[1]);
            MPI_Isend(local + my_rows * cols,    cols, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(local + (my_rows + 1)*cols,cols, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[3]);
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        } 
        /* 2. Communication Strategy: Blocking/Deadlock-Safe (Odd steps) */
        else {
            /* 
             * DEADLOCK SCENARIO (Commented for documentation):
             * if (rank % 2 == 0) { MPI_Send(..., up, ...); MPI_Recv(..., down, ...); }
             * else               { MPI_Send(..., down, ...); MPI_Recv(..., up, ...); }
             * Why: If buffers fill, both blocks wait forever.
             * Fix: Use MPI_Sendrecv (atomic send+recv, guaranteed progress)
             */
            MPI_Sendrecv(local + cols,              cols, MPI_DOUBLE, up,   0,
                         local + (my_rows+1)*cols,  cols, MPI_DOUBLE, down, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(local + my_rows * cols,    cols, MPI_DOUBLE, down, 1,
                         local,                     cols, MPI_DOUBLE, up,   1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        /* 1. Stencil Computation (5-point Laplacian) */
        double r_factor = ALPHA * DT / (DX * DX);
        for (int lr = 1; lr <= my_rows; ++lr) {
            int gr = displs[rank] / cols + (lr - 1);
            for (int c = 0; c < cols; ++c) {
                /* Boundary conditions fixed */
                if (gr == 0 || gr == rows - 1 || c == 0 || c == cols - 1) {
                    local_new[lr * cols + c] = local[lr * cols + c];
                    continue;
                }
                double center = local[lr * cols + c];
                double top    = local[(lr-1)*cols + c];
                double bot    = local[(lr+1)*cols + c];
                double lft    = local[lr * cols + c - 1];
                double rgt    = local[lr * cols + c + 1];
                local_new[lr * cols + c] = center + r_factor * (top + bot + lft + rgt - 4.0 * center);
            }
        }
        memcpy(local, local_new, local_size * sizeof(double));
    }

    double elapsed = MPI_Wtime() - t_start;

    /* Collect results */
    if (rank == 0) global_grid = (double *)realloc(global_grid, rows * cols * sizeof(double));
    MPI_Gatherv(local + cols, my_rows * cols, MPI_DOUBLE,
                rank == 0 ? global_grid : NULL, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* 6. Performance Analysis */
    double max_time;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[Heat Diffusion] Grid=%dx%d | Steps=%d | Procs=%d | Time=%.4fs\n", 
               rows, cols, steps, size, max_time);
        save_grid(global_grid, rows, cols, "heat_result.csv");
        printf(" -> Saved heat_result.csv\n");
        free(global_grid);
    }

    free(sendcounts); free(displs); free(local); free(local_new);
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
    run_heat_diffusion(rank, size);
    MPI_Finalize();
    return 0;
}