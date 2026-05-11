// =====================================================================
// Deadlock Demonstration & Fix
// =====================================================================
// Scenario:
//   Two processes (rank 0 and rank 1) each call MPI_Send FIRST and then
//   MPI_Recv. If MPI_Send is BLOCKING and the message is too large to be
//   buffered by the MPI implementation, both processes will wait forever
//   for the other side to start receiving -> classic deadlock.
//
// Why it happens:
//   - Both calls are blocking
//   - Both processes try to send before either of them is ready to receive
//   - MPI cannot proceed because the matching receives never get posted
//
// When it happens:
//   - Large messages (above the eager-protocol threshold) almost always
//     cause this. Small ones may "work" by luck because MPI buffers them.
//
// FIX: reverse the order on one side (or use MPI_Sendrecv / non-blocking).
// =====================================================================

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

void runDeadlockDemo(int rank, int size) {
    if (size < 2) {
        if (rank == 0)
            std::cerr << "[Deadlock] Need at least 2 processes.\n";
        return;
    }
    // Only ranks 0 and 1 participate; others wait at the barrier.
    if (rank == 0) {
        std::cout << "\n--- [Deadlock Demonstration] ---\n";
        std::cout << "Step 1: showing the BROKEN version (commented out in code)\n";
        std::cout << "        Each rank does Send then Recv with a LARGE message.\n";
        std::cout << "        With MPI_Send blocking, this hangs forever.\n\n";
        std::cout << "Step 2: showing the FIXED version using MPI_Sendrecv ...\n";
    }

    const int N = 1000000;       // Large enough to bypass eager buffering
    std::vector<int> sendBuf(N, rank);
    std::vector<int> recvBuf(N, 0);

    /* -----------------------------------------------------------------
       BROKEN VERSION (kept commented so the demo doesn't hang):

       if (rank == 0 || rank == 1) {
           int partner = 1 - rank;
           MPI_Send(sendBuf.data(), N, MPI_INT, partner, 0, MPI_COMM_WORLD);
           MPI_Recv(recvBuf.data(), N, MPI_INT, partner, 0, MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
       }
       ----------------------------------------------------------------- */

    // ------------------- FIXED VERSION ---------------------------------
    // Option A: MPI_Sendrecv (atomic, deadlock-free)
    if (rank == 0 || rank == 1) {
        int partner = 1 - rank;
        MPI_Sendrecv(sendBuf.data(), N, MPI_INT, partner, 0,
                     recvBuf.data(), N, MPI_INT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Rank " << rank << " exchanged data with rank "
                  << partner << " (recvBuf[0] = " << recvBuf[0] << ")\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n[Deadlock Demo] Successfully completed using MPI_Sendrecv.\n";
        std::cout << "Alternative fix: use MPI_Isend/MPI_Irecv with MPI_Waitall.\n";
    }
}
