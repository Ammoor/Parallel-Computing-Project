// =====================================================================
// Heat Diffusion (2D Stencil) - Parallel Implementation with MPI
// =====================================================================
//  - 1D row-based decomposition of a 2D grid.
//  - Supports UNEVEN data distribution (rows not divisible by P).
//  - Three communication modes: blocking | nonblocking | collective.
//  - Boundary conditions: Dirichlet (fixed-edge values from input).
//  - Stencil: 4-point average  T_new[i][j] = (T[i-1][j]+T[i+1][j]+T[i][j-1]+T[i][j+1]) / 4
//  - Uses MPI_Comm_split to create a "workers" sub-communicator (rank > 0
//    can also participate; here we split by even/odd rank to demonstrate
//    logical process grouping for structured halo exchange).
// =====================================================================

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstring>
#include <iomanip>

static void loadGrid(const std::string& path, int& R, int& C,
                     std::vector<double>& grid) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "[Heat] Cannot open input file: " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    fin >> R >> C;
    grid.assign(R * C, 0.0);
    for (int i = 0; i < R * C; ++i) fin >> grid[i];
}

void runHeatDiffusion(int argc, char** argv, int rank, int size) {
    // ---- Parse arguments ----
    std::string inputFile = "data/heat_input.txt";
    int iterations = 100;
    std::string mode = "collective";   // blocking | nonblocking | collective

    if (argc >= 3) inputFile  = argv[2];
    if (argc >= 4) iterations = std::atoi(argv[3]);
    if (argc >= 5) mode       = argv[4];

    if (rank == 0) {
        std::cout << "\n--- [Heat Diffusion] ---\n";
        std::cout << "Input file : " << inputFile  << "\n";
        std::cout << "Iterations : " << iterations << "\n";
        std::cout << "Comm mode  : " << mode       << "\n";
        std::cout << "Processes  : " << size       << "\n";
    }

    // ---- Load grid on root and broadcast dimensions ----
    int R = 0, C = 0;
    std::vector<double> fullGrid;
    if (rank == 0) loadGrid(inputFile, R, C, fullGrid);
    MPI_Bcast(&R, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&C, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- Uneven row distribution ----
    std::vector<int> rowsPerProc(size), displsRows(size);
    int base = R / size, extra = R % size;
    for (int i = 0; i < size; ++i) {
        rowsPerProc[i] = base + (i < extra ? 1 : 0);
    }
    displsRows[0] = 0;
    for (int i = 1; i < size; ++i)
        displsRows[i] = displsRows[i - 1] + rowsPerProc[i - 1];

    int localRows = rowsPerProc[rank];

    // Sendcounts / displs in elements (rows * C) for Scatterv/Gatherv
    std::vector<int> sendCounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        sendCounts[i] = rowsPerProc[i] * C;
        displs[i]     = displsRows[i] * C;
    }

    // Local block has 2 ghost (halo) rows: row 0 and row localRows+1.
    std::vector<double> local((localRows + 2) * C, 0.0);
    std::vector<double> newLocal((localRows + 2) * C, 0.0);

    // Scatter rows to all processes
    MPI_Scatterv(rank == 0 ? fullGrid.data() : nullptr,
                 sendCounts.data(), displs.data(), MPI_DOUBLE,
                 &local[C],          // skip ghost row 0
                 localRows * C, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // ---- MPI_Comm_split: organize processes (even/odd) into groups -----
    // Demonstrates "process organization into logical groups".
    int color = rank % 2;
    MPI_Comm groupComm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &groupComm);
    int groupRank, groupSize;
    MPI_Comm_rank(groupComm, &groupRank);
    MPI_Comm_size(groupComm, &groupSize);
    if (rank == 0) {
        std::cout << "MPI_Comm_split: created 2 groups (even/odd ranks)\n";
        std::cout << "Group 0 size = " << ((size + 1) / 2)
                  << ", Group 1 size = " << (size / 2) << "\n";
    }

    int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;

    double t0 = MPI_Wtime();

    // ---- Iteration loop ----
    for (int it = 0; it < iterations; ++it) {

        // ====== Halo exchange (depending on chosen mode) ======
        if (mode == "blocking") {
            // Send/Recv pairs - ordered to avoid deadlock
            // Even ranks send up first, odd ranks recv from down first
            if (rank % 2 == 0) {
                MPI_Send(&local[1 * C],            C, MPI_DOUBLE, up,   0, MPI_COMM_WORLD);
                MPI_Recv(&local[0],                C, MPI_DOUBLE, up,   1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&local[localRows * C],    C, MPI_DOUBLE, down, 2, MPI_COMM_WORLD);
                MPI_Recv(&local[(localRows+1)*C],  C, MPI_DOUBLE, down, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(&local[(localRows+1)*C],  C, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&local[localRows * C],    C, MPI_DOUBLE, down, 1, MPI_COMM_WORLD);
                MPI_Recv(&local[0],                C, MPI_DOUBLE, up,   2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(&local[1 * C],            C, MPI_DOUBLE, up,   3, MPI_COMM_WORLD);
            }
        }
        else if (mode == "nonblocking") {
            MPI_Request reqs[4];
            MPI_Isend(&local[1 * C],            C, MPI_DOUBLE, up,   0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(&local[localRows * C],    C, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &reqs[1]);
            MPI_Irecv(&local[0],                C, MPI_DOUBLE, up,   1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Irecv(&local[(localRows+1)*C],  C, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &reqs[3]);
            MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
        }
        else { // collective: gather boundary rows from all procs, scatter back the needed neighbors
            // Each process contributes its TOP and BOTTOM rows to an Allgather.
            // Then every process picks neighbor rows from the gathered buffer.
            std::vector<double> myEdges(2 * C);
            std::memcpy(&myEdges[0],    &local[1 * C],         C * sizeof(double));
            std::memcpy(&myEdges[C],    &local[localRows * C], C * sizeof(double));

            std::vector<double> allEdges(2 * C * size);
            MPI_Allgather(myEdges.data(),  2 * C, MPI_DOUBLE,
                          allEdges.data(), 2 * C, MPI_DOUBLE, MPI_COMM_WORLD);

            // up neighbor's BOTTOM row = allEdges[(rank-1)*2*C + C ... + 2C]
            if (up != MPI_PROC_NULL) {
                std::memcpy(&local[0], &allEdges[(rank - 1) * 2 * C + C],
                            C * sizeof(double));
            }
            // down neighbor's TOP row = allEdges[(rank+1)*2*C + 0 ... + C]
            if (down != MPI_PROC_NULL) {
                std::memcpy(&local[(localRows + 1) * C],
                            &allEdges[(rank + 1) * 2 * C], C * sizeof(double));
            }
        }

        // ====== Stencil computation ======
        for (int i = 1; i <= localRows; ++i) {
            for (int j = 0; j < C; ++j) {
                // Keep boundary (left/right column, very top of grid, very bottom of grid) fixed
                bool isGlobalEdge =
                    (rank == 0        && i == 1)              ||
                    (rank == size - 1 && i == localRows)      ||
                    (j == 0) || (j == C - 1);

                if (isGlobalEdge) {
                    newLocal[i * C + j] = local[i * C + j];
                } else {
                    newLocal[i * C + j] = 0.25 * (
                        local[(i - 1) * C + j] +
                        local[(i + 1) * C + j] +
                        local[i * C + (j - 1)] +
                        local[i * C + (j + 1)]);
                }
            }
        }
        std::swap(local, newLocal);
    }

    double t1 = MPI_Wtime();

    // ---- Gather result back to root ----
    std::vector<double> result;
    if (rank == 0) result.assign(R * C, 0.0);
    MPI_Gatherv(&local[C], localRows * C, MPI_DOUBLE,
                rank == 0 ? result.data() : nullptr,
                sendCounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        double elapsed = t1 - t0;
        std::cout << "\n[Heat Diffusion] Done.\n";
        std::cout << "Grid: " << R << "x" << C
                  << ", Iter: " << iterations
                  << ", Mode: " << mode << "\n";
        std::cout << "Elapsed time : " << std::fixed << std::setprecision(6)
                  << elapsed << " s\n";

        // Print a small preview (first 5x5)
        int pr = std::min(R, 5), pc = std::min(C, 5);
        std::cout << "Result preview (top-left " << pr << "x" << pc << "):\n";
        std::cout << std::fixed << std::setprecision(3);
        for (int i = 0; i < pr; ++i) {
            for (int j = 0; j < pc; ++j)
                std::cout << std::setw(8) << result[i * C + j] << " ";
            std::cout << "\n";
        }

        // Save full result
        std::ofstream fout("data/heat_output.txt");
        fout << R << " " << C << "\n";
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j)
                fout << result[i * C + j] << (j == C - 1 ? '\n' : ' ');
        }
        std::cout << "Full result written to data/heat_output.txt\n";
    }

    MPI_Comm_free(&groupComm);
}
