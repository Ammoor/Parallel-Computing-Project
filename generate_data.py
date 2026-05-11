#!/usr/bin/env python3
"""
Generate input data files for the PC 2026 project.
Run once before executing the MPI program:
    python3 generate_data.py
"""
import random, os
random.seed(42)

os.makedirs("data", exist_ok=True)

# ---------- Heat diffusion grid (e.g. 200x200) ----------
R, C = 200, 200
with open("data/heat_input.txt", "w") as f:
    f.write(f"{R} {C}\n")
    for i in range(R):
        row = []
        for j in range(C):
            # Hot top row, cold elsewhere, plus a hot spot in the middle
            if i == 0:
                row.append(100.0)
            elif i == R - 1:
                row.append(0.0)
            elif j == 0 or j == C - 1:
                row.append(50.0)
            elif R // 2 - 5 <= i <= R // 2 + 5 and C // 2 - 5 <= j <= C // 2 + 5:
                row.append(75.0)
            else:
                row.append(20.0)
        f.write(" ".join(f"{v:.2f}" for v in row) + "\n")
print(f"Wrote data/heat_input.txt  ({R}x{C} grid)")

# ---------- Prefix sum array (e.g. 1,000,000 elements) ----------
N = 1_000_000
with open("data/prefix_input.txt", "w") as f:
    f.write(f"{N}\n")
    for _ in range(N):
        f.write(f"{random.randint(1, 10)}\n")
print(f"Wrote data/prefix_input.txt ({N} integers)")
