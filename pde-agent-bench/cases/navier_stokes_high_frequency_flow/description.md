# Case: navier_stokes_high_frequency_flow

## Problem Description

Solve the Poisson equation on a unit square domain [0,1]×[0,1]:

  -∇·(κ ∇u) = f   in Ω
  u = g           on ∂Ω

This is an elliptic boundary value problem.

**Problem Parameters:**
- Manufactured solution: u = ['2*pi*cos(2*pi*y)*sin(2*pi*x)', '-2*pi*cos(2*pi*x)*sin(2*pi*y)']
- Source term f and boundary data g are derived from the manufactured solution

**Boundary Conditions:**
- Dirichlet BC on all boundaries: u = u_exact (from manufactured solution)

**Requirements:**
Your implementation must:
1. Use `dolfinx` (FEniCSx) for finite element assembly and solving
2. **YOU decide the optimal discretization parameters** (mesh resolution, element degree, etc.)
3. Accept command-line argument: `--outdir DIR` (output directory)
4. Save the solution to `solution.npz` with fields: `x` (1D array), `y` (1D array), `u` (2D array)
5. Save solver metadata to `meta.json` with fields:
   - `wall_time_sec`: Your solver runtime (required)
   - `solver_info`: Dictionary with (required):
     * `mesh_resolution`: The resolution you chose (e.g., 200)
     * `element_degree`: The polynomial degree you chose (e.g., 1 or 2)
     * `ksp_type`: Linear solver type (e.g., 'cg', 'gmres')
     * `pc_type`: Preconditioner type (e.g., 'hypre', 'ilu')
     * `rationale`: (optional) Brief explanation of your parameter choices

**Output Grid:**
Sample the solution on a uniform 50×50 grid spanning the domain.

**Parameter Selection Strategy:**
You must balance accuracy and speed:
- Higher resolution → better accuracy, slower runtime
- Higher element degree → better accuracy per DOF, more expensive assembly
- Choose wisely based on the target error and time budget!

---

## Testing Modes

This case supports two testing modes:

### Mode 1: Fix Accuracy, Optimize Speed
**Goal:** Meet accuracy requirement in minimum time.

```bash
python test_fix_accuracy.py --agent-script your_solver.py
```

**Challenge:** Choose resolution/degree to achieve the target error as fast as possible.

**Scoring:**
- Must achieve `error` ≤ 0.01
- Score = 100 × (time_budget / your_runtime)
- Faster = Higher score
- Smart parameter choices = Better scores!

### Mode 2: Fix Time Budget, Optimize Accuracy
**Goal:** Achieve best accuracy within time budget.

```bash
python test_fix_time.py --agent-script your_solver.py
```

**Challenge:** Choose resolution/degree to minimize error while staying within time budget.

**Scoring:**
- Must finish within time_budget
- Score = 100 × max(0, 1 - error/target_error)
- Lower error = Higher score
- Efficient solvers = Better scores!

---

## Target Metrics

- Target Error: 0.01
- Timeout: 300 seconds

---

## Difficulty Tiers

This case provides **3-level evaluation** to comprehensively assess your solver's quality:

### Accuracy Levels (Fix Time Budget, Optimize Accuracy)
- **Level 1 (Engineering Grade)**: Target error ≤ 1.00e+00
  * Relaxed requirement (100× Oracle baseline)
  * Suitable for rough estimates
  
- **Level 2 (Standard Grade)**: Target error ≤ 1.00e-02 ⭐ **Oracle baseline**
  * Default requirement
  * Represents good numerical practice
  
- **Level 3 (Scientific Grade)**: Target error ≤ 1.00e-04
  * Strict requirement (0.01× Oracle baseline)
  * Requires refined meshes or high-order elements

### Speed Levels (Fix Accuracy, Optimize Speed)
- **Level 3 (Real-time)**: Time budget = 1.000s (0.1× Oracle)
  * Extremely fast (requires optimal algorithms)
  
- **Level 2 (Interactive)**: Time budget = 10.000s ⭐ **Oracle baseline**
  * Default requirement
  * Standard solver performance
  
- **Level 1 (Batch)**: Time budget = 100.000s (10× Oracle)
  * Relaxed requirement (suitable for direct solvers)

**Evaluation Strategy:**
Your solver will be scored based on how many levels you pass:
- Passing Level 2 = Good solver (meets standard requirements)
- Passing Level 3 = Excellent solver (superior numerical methods or parameter choices)

**Oracle Reference:**
The Oracle baseline was obtained with optimized parameters. Your goal is to match or exceed this performance through smart parameter selection and efficient algorithms.
