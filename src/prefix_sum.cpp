// =====================================================================
// Parallel Prefix Sum (Inclusive Scan) using MPI
// =====================================================================
//  - Loads a large 1D array from a file.
//  - Distributes elements across processes using Scatterv (uneven OK).
//  - Each process computes local prefix sum.
//  - Uses MPI_Exscan (COLLECTIVE-based communication) to obtain the
//    offset from all previous processes, then adds it to local prefix.
//  - Final result is gathered to root using Gatherv.
//  - Also uses MPI_Comm_split to demonstrate logical process grouping.
// =====================================================================

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <iomanip>

static void loadArray(const std::string& path, int& N,
                      std::vector<long long>& arr) {
    std::ifstream fin(path);
    if (!fin) {
        std::cerr << "[Prefix] Cannot open input file: " << path << "\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
    }
    fin >> N;
    arr.assign(N, 0);
    for (int i = 0; i < N; ++i) fin >> arr[i];
}

void runPrefixSum(int /*argc*/, char** argv, int rank, int size) {
    std::string inputFile = "data/prefix_input.txt";
    if (argv[2] != nullptr) inputFile = argv[2];

    if (rank == 0) {
        std::cout << "\n--- [Prefix Sum] ---\n";
        std::cout << "Input file : " << inputFile << "\n";
        std::cout << "Processes  : " << size      << "\n";
    }

    // ---- Load data on root ----
    int N = 0;
    std::vector<long long> data;
    if (rank == 0) loadArray(inputFile, N, data);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- Uneven distribution ----
    std::vector<int> counts(size), displs(size);
    int base = N / size, extra = N % size;
    for (int i = 0; i < size; ++i)
        counts[i] = base + (i < extra ? 1 : 0);
    displs[0] = 0;
    for (int i = 1; i < size; ++i)
        displs[i] = displs[i - 1] + counts[i - 1];

    int localN = counts[rank];
    std::vector<long long> localArr(localN, 0);

    // ---- MPI_Comm_split: group processes by half (first half / second half)
    int color = (rank < size / 2) ? 0 : 1;
    MPI_Comm groupComm;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &groupComm);
    int groupSize;
    MPI_Comm_size(groupComm, &groupSize);
    if (rank == 0) {
        std::cout << "MPI_Comm_split: split into 2 logical groups\n";
        std::cout << "  Group A (lower half) : " << (size / 2 + size % 2) << " procs\n";
        std::cout << "  Group B (upper half) : " << (size / 2) << " procs\n";
    }

    double t0 = MPI_Wtime();

    // ---- COLLECTIVE Scatterv: distribute the data ----
    MPI_Scatterv(rank == 0 ? data.data() : nullptr,
                 counts.data(), displs.data(), MPI_LONG_LONG,
                 localArr.data(), localN, MPI_LONG_LONG,
                 0, MPI_COMM_WORLD);

    // ---- Step 1: local inclusive prefix sum ----
    std::vector<long long> localPrefix(localN, 0);
    long long running = 0;
    for (int i = 0; i < localN; ++i) {
        running += localArr[i];
        localPrefix[i] = running;
    }
    long long localTotal = running;

    // ---- Step 2: MPI_Exscan to get sum of ALL previous processes ----
    // This is the COLLECTIVE-based communication pattern.
    long long offset = 0;
    MPI_Exscan(&localTotal, &offset, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) offset = 0;  // Exscan undefined at rank 0

    // ---- Step 3: add offset to every local prefix value ----
    for (int i = 0; i < localN; ++i)
        localPrefix[i] += offset;

    // ---- COLLECTIVE Gatherv: assemble full result ----
    std::vector<long long> result;
    if (rank == 0) result.assign(N, 0);
    MPI_Gatherv(localPrefix.data(), localN, MPI_LONG_LONG,
                rank == 0 ? result.data() : nullptr,
                counts.data(), displs.data(), MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "\n[Prefix Sum] Done.\n";
        std::cout << "N = " << N << ", Elapsed = "
                  << std::fixed << std::setprecision(6) << (t1 - t0) << " s\n";

        // Print first / last few entries
        int show = std::min(N, 10);
        std::cout << "First " << show << " input values   : ";
        for (int i = 0; i < show; ++i) std::cout << data[i] << " ";
        std::cout << "\nFirst " << show << " prefix-sum values: ";
        for (int i = 0; i < show; ++i) std::cout << result[i] << " ";
        std::cout << "\nTotal sum (last element) = " << result[N - 1] << "\n";

        // Save
        std::ofstream fout("data/prefix_output.txt");
        fout << N << "\n";
        for (int i = 0; i < N; ++i) fout << result[i] << "\n";
        std::cout << "Full result written to data/prefix_output.txt\n";
    }

    MPI_Comm_free(&groupComm);
}
