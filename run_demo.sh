#!/bin/bash
# =====================================================================
# Demo script - runs all parts of the project with mpiexec
# =====================================================================
set -e

echo "================================================="
echo " Step 1: Generating input data files"
echo "================================================="
python3 generate_data.py

echo
echo "================================================="
echo " Step 2: Building the project (make)"
echo "================================================="
make clean
make

echo
echo "================================================="
echo " Step 3: Heat Diffusion - blocking communication"
echo "================================================="
mpiexec -n 4 ./pc_project 1 data/heat_input.txt 100 blocking

echo
echo "================================================="
echo " Step 4: Heat Diffusion - non-blocking comm"
echo "================================================="
mpiexec -n 4 ./pc_project 1 data/heat_input.txt 100 nonblocking

echo
echo "================================================="
echo " Step 5: Heat Diffusion - COLLECTIVE comm"
echo "================================================="
mpiexec -n 4 ./pc_project 1 data/heat_input.txt 100 collective

echo
echo "================================================="
echo " Step 6: Prefix Sum (collective Scatterv/Exscan/Gatherv)"
echo "================================================="
mpiexec -n 4 ./pc_project 2 data/prefix_input.txt

echo
echo "================================================="
echo " Step 7: Deadlock demonstration + fix"
echo "================================================="
mpiexec -n 2 ./pc_project 3

echo
echo "================================================="
echo " Step 8: Scalability run for Heat Diffusion (collective)"
echo "================================================="
for P in 1 2 4 8; do
  echo "--- mpiexec -n $P ---"
  if [ "$P" -eq 1 ]; then
    # main.cpp enforces N>=2, so skip P=1 (or use --oversubscribe with N=2)
    echo "(skipped, project requires N>=2)"
  else
    mpiexec -n $P ./pc_project 1 data/heat_input.txt 100 collective \
        | grep -E "Elapsed|Processes"
  fi
done

echo
echo "================================================="
echo " All demos finished successfully."
echo "================================================="
