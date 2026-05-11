// =====================================================================
// Advanced Parallel Grid Processing with MPI
// Faculty of Computers and Artificial Intelligence - Fayoum University
// Parallel Computing - Practical Project (2026)
// =====================================================================
// Entry point: lets the user select an algorithm at runtime.
//   1) Heat Diffusion (Category A - Stencil)
//   2) Prefix Sum     (Category B - Scan)
//   3) Deadlock Demo  (intentional + fixed version)
// =====================================================================

#include <mpi.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>

// Forward declarations
void runHeatDiffusion(int argc, char** argv, int rank, int size);
void runPrefixSum(int argc, char** argv, int rank, int size);
void runDeadlockDemo(int rank, int size);

static void printMenu() {
    std::cout << "\n==================================================\n";
    std::cout << " Advanced Parallel Grid Processing with MPI (2026)\n";
    std::cout << "==================================================\n";
    std::cout << " 1) Heat Diffusion  (Category A - Stencil)\n";
    std::cout << " 2) Prefix Sum      (Category B - Scan)\n";
    std::cout << " 3) Deadlock Demo   (intentional + fix)\n";
    std::cout << "==================================================\n";
    std::cout << "Usage: mpiexec -n <P> ./pc_project <choice> [args...]\n";
    std::cout << "  Heat:   mpiexec -n 4 ./pc_project 1 data/heat_input.txt 100 blocking\n";
    std::cout << "          comm modes: blocking | nonblocking | collective\n";
    std::cout << "  Prefix: mpiexec -n 4 ./pc_project 2 data/prefix_input.txt\n";
    std::cout << "  Dead:   mpiexec -n 2 ./pc_project 3\n";
    std::cout << "==================================================\n\n";
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0)
            std::cerr << "[Error] This project requires N >= 2 processes.\n";
        MPI_Finalize();
        return 1;
    }

    int choice = 0;
    if (argc >= 2) {
        choice = std::atoi(argv[1]);
    } else {
        if (rank == 0) {
            printMenu();
            std::cout << "Enter choice (1/2/3): ";
            std::cout.flush();
            std::cin >> choice;
        }
        MPI_Bcast(&choice, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    switch (choice) {
        case 1: runHeatDiffusion(argc, argv, rank, size); break;
        case 2: runPrefixSum(argc, argv, rank, size);    break;
        case 3: runDeadlockDemo(rank, size);             break;
        default:
            if (rank == 0) {
                std::cerr << "[Error] Invalid choice. Use 1, 2, or 3.\n";
                printMenu();
            }
    }

    MPI_Finalize();
    return 0;
}
